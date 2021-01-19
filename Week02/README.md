学习笔记

1.把提供的 target encoding 代码改为 cython 代码并比较速度区别（如可以实现并行可加分）
[target encoding 代码获取地址]： https://github.com/rwbfd/ml-training-camp/tree/main/chap02

2.附加题：
查看 B-spline 的介绍。
使用 Cython 实现对输入多列返回 B-spline basis 的操
作。
注意：禁止使用函数 recursive call。
注意：必须要处理异常情况，例如缺失值、inf 等

# Notes

Please run `testing.py` to see the results.

```
target_mean_v1 took 28960.182091 ms
target_mean_v2 took 347.30334500000026 ms
target_mean_v3 took 1.3552800000020682 ms
target_mean_v4 took 1.3159369999975468 ms
target_mean_v5 took 0.8711639999994247 ms
```

```
v1 is the unoptimized version.
v2 is the two loops version.
v3 just converts the arguments of the function to cython.
v4 is cython version but has used python type `dict()`
v5 is completely cython version.
```

## TODO

- [ ] replace the type of `map`
- [ ] use multi-threaded