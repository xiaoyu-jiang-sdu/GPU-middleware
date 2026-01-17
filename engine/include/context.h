#pragma once

namespace engine {

class Context {
public:
    virtual ~Context() = default;

    // 同步流
    virtual void sync() = 0;
};

} // namespace engine