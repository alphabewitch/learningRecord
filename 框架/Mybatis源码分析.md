# Mybatis源码分析

## 一、概述

我们在开发中通过MyBatis来连接操作数据库，配置好MyBatis之后，通过自定义的Mapper接口和对应的SQL语句来便捷地增删改查。那么为什么我们可以直接通过一个没有实现类的接口来实现操作呢，MyBatis在背后帮我们做了什么呢，我一直比较好奇。首先来看下怎么使用MyBatis。

### 1、项目构建

1. 添加Maven依赖，引入jar包

![image-20230504144728574](img/Mybatis源码分析/image-20230504144728574.png)

2. 编写MyBatis-config.xml核心配置文件

![image-20230504145015093](img/Mybatis源码分析/image-20230504145015093.png)

3. 写好实体类

![image-20230504145100199](img/Mybatis源码分析/image-20230504145100199.png)

4. 写好Mapper/Dao接口

![image-20230504145122052](img/Mybatis源码分析/image-20230504145122052.png)

5. 写Mapper对应的XML的SQL

![image-20230504145138528](img/Mybatis源码分析/image-20230504145138528.png)

