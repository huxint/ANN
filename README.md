# ANN — 基于 C++23 的简易神经网络小框架

## 简介

这是一个使用 **C++23** 编写的极简人工神经网络库：

* 仅依赖 **Eigen** 进行矩阵与向量运算
* 使用 **xmake** 作为构建工具

> 本项目的目标是**学习与理解神经网络的原理**，
> 不追求性能、工程鲁棒性或完整的特性集。

## 构建方式

项目使用 [**xmake**](https://xmake.io) 构建。

在项目根目录下执行：

```bash
xmake
xmake run ANN
```

## 依赖项

* **xmake**
* **Eigen** 库（仅头文件）

### 安装 Eigen

1. 从 Eigen 官网下载最新版本：

   * 官网：[https://eigen.tuxfamily.org/index.php?title=Main_Page](https://eigen.tuxfamily.org/index.php?title=Main_Page)
   * 或直接下载： [eigen-5.0.0.tar.bz2](https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.bz2)

2. 解压后，在项目根目录创建一个 `third_party` 文件夹，并将解压后的 `eigen` 目录放进去，结构大致如下：

   ```txt
   third_party/
       eigen/
           Eigen/
           unsupported/
           CMakeLists.txt
           ...
   ```

3. `xmake.lua` 中已经配置了包含路径（例如 `add_includedirs("third_party/eigen")` 或类似配置），确保路径与实际目录一致即可。