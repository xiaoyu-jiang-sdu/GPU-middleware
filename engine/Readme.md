# Engine 
Engine/ 实现C++层GPU算子的调用实现，并为上层py提供规定的接口。
Engine/ 需要实现tensor、context、engine三大内容。
具体而言：
- tensor: 张量在GPU上存储形式，需要提供数据指针、张量形状、strides等元信息
- context: 用于管理GPU上下文，包括句柄的创建销毁、引用对象的管理
- engine: 封装并实现算子，为上层adapter提供功能调用

以上为所有GPU实现基于ONNX的模型推理的基类。