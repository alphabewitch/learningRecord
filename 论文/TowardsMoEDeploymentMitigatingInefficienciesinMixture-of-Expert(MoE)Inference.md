# Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

## 01 贡献

1. 对MoE模型的部署进行全面分解分析，针对推理延迟和内存使用情况来识别效率低下的根源。
2. 认为Gate是造成MoE模型高延迟和大内存占用的主要原因，提出新颖的Gate策略，能显著降低推理延迟和内存占用，能使用更大的batch size和更少的GPU数量。
3. 发现专家之间负载分配不均匀，时间局部性较高。提出Expert Buffer，仅将热点或者活动专家保留在GPU内存中，将其他专家移到CPU主存。
4. 提出平衡专家之间负载的技术，提升内存使用和系统鲁棒性。

![image-20230725202455140](img/TowardsMoEDeploymentMitigatingInefficienciesinMixture-of-Expert(MoE)Inference/image-20230725202455140.png)

## 02 MoE模型特征

将MoE模型和相应的稠密Transformer模型对比，选择语言生成和机器翻译两个任务，来分析MoE模型的特征。

### 1、延迟和存储

作者对比稠密Transformer和相应的MoE模型，将不同部分产生的延迟进行拆分

![image-20230726100617394](img/TowardsMoEDeploymentMitigatingInefficienciesinMixture-of-Expert(MoE)Inference/image-20230726100617394.png)

在多节点情况下All-to-all通信占据了延迟的大部分，但是我们可以发现还有其他组件也会产生很多延迟值得优化。

作者将内存分为静态和动态，静态内存消耗是指分配给模型参数的内存，动态内存指按需内配的内存。因为每个GPU要保存更多的专家，所以静态内存显然是会增加。但是观察到动态内存也会显著增加。

### 2、静态Gating效率低

之前的MoE都是使用的静态Gating，token分配过程被简化为分配相同数量的token，all-to-all到所有设备。

定义容量因子C表示每个专家在一个batch中处理的token数量。如果Gating分配的token数量少于专家容量C，则会用占位符填充。如果分配多余C，则会将多余的token丢弃，但是丢弃token会损害精度。所以C通常设定得比较大，这会导致专家处理过多的token，也会导致增加通信成本和存储成本。

如果工作负载能得到平衡，那么就可以减少C，来减少资源浪费。

## 03 Expert激活模式分析

作者针对语言生成和翻译这两个任务，可视化他们的专家激活情况。下图中，每行代表一个batch，每列代表特定专家的负载，颜色越深表示专家收到的token比例越高，发现专家之间负载高度不平衡。

![image-20230726152048895](img/TowardsMoEDeploymentMitigatingInefficienciesinMixture-of-Expert(MoE)Inference/image-20230726152048895.png)

但是由于静态Gating，有的专家就算没有收到过token，也会收到和处理空的占位符，造成浪费。

作者还发现针对不同的任务和数据集，专家的热门情况有很大差别。**非常重要的是发现专家激活具有很强的时间局部性。高激活的专家颜色深且成为线条状，表明专家连续处于激活状态。**

## 04 动态Gating优化

通过上述的分析，作者的结论是，静态Gating增加了资源浪费，固定的专家容量不是分配给专家的令牌的最优解决方案。所以要设计动态Gating。

下图a中显示静态Gating的决策过程，生成Gating Decision对应的Dispatch Mask，输入的Token根据Mask确定要被分配到哪个设备上。

![image-20230726162211685](img/TowardsMoEDeploymentMitigatingInefficienciesinMixture-of-Expert(MoE)Inference/image-20230726162211685.png)

上图b中是重新设计的动态Gating。先同样生成Gating Decision，因为没有大小限制，直接得到每个token被分到哪个设备上。此外可以排序Gating Decision得到每个设备要分派token的size。在传输时先通知每个设备要传入的token size，再使用All-to-all真正传输token。这种动态Gating不会在设备之间传输空占位符，减少通信和计算开销。

![image-20230726173309969](img/TowardsMoEDeploymentMitigatingInefficienciesinMixture-of-Expert(MoE)Inference/image-20230726173309969.png)

上图中蓝色、橘色、绿色分别是静态Gating、Tutel Gating、动态Gating，发现能显著提高所有batch大小和任务情况下的吞吐率。

![image-20230726173551394](img/TowardsMoEDeploymentMitigatingInefficienciesinMixture-of-Expert(MoE)Inference/image-20230726173551394.png)

上图中显示了单节点的情况下不同Gating策略对内存消耗的影响。因为动态Gating不用Mask，删掉了空填充和占位符，减少的内存的浪费分配。可以看到动态Gating能减少非常多的内存占用。

## 05 Expert Buffer

使用动态Gating还留下了一个弊端，专家激活模式存在高度稀疏型，有的专家很热门，有的专家很冷门，所以作者想修剪掉空闲的专家来减少GPU内存的占用。可以将访问频率较低的专家卸载到CPU内存，热门专家保留在GPU内存。

动态Gating做出决策之后，每个专家能收到将要收到token的size，如果size是正的，说明要激活这个expert，这时检查它是否在GPU内存中，如果不存在，就要从CPU内存中加载它到GPU内存。并且可以在当前层做计算时提前检查下一层做到通信和计算的重叠。

如果缓存已满但需要更多专家，则会触发驱逐以为新专家腾出空间。驱逐策略设计如下：将⾸先驱逐在该批次中不活跃的专家，因为由于时间局部性，他们将来也不太可能被使⽤。将根据**后进先出 (LIFO)** 策略逐出专家参数。

![image-20230726201021794](img/TowardsMoEDeploymentMitigatingInefficienciesinMixture-of-Expert(MoE)Inference/image-20230726201021794.png)

作者计算了每个设备存储不同数量专家时的缓存命中率，发现当size小于5时，miss率上升很快。图b中显示LIFO的效果比FIFO要更好。

在上一节的图9、图10中表明Expert Buffer能减少显存的占用，但是吞吐量会减少。

![image-20230726203204767](img/TowardsMoEDeploymentMitigatingInefficienciesinMixture-of-Expert(MoE)Inference/image-20230726203204767.png)

由于这个Expert Buffer涉及到GPU显存和CPU主存之间的换入换出，这之间也会产生延迟，且这个延迟主要取决于两者之间的带宽。

## 06 负载均衡

热门专家的GPU会成为瓶颈，可能OOM，但是冷门专家的GPU可能空闲。













