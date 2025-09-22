#!/bin/bash

# 批量生成所有场景的脚本（CUDA版本）
# 用法: ./cugen.sh

echo "开始批量生成所有场景（使用CUDA渲染器）..."

# 创建CUDA输出文件夹
CUDA_OUTPUT_DIR="cuda_output"
echo "创建CUDA输出文件夹: $CUDA_OUTPUT_DIR"
if [ ! -d "$CUDA_OUTPUT_DIR" ]; then
    mkdir -p "$CUDA_OUTPUT_DIR"
    echo "已创建CUDA输出文件夹"
else
    echo "CUDA输出文件夹已存在"
fi

# 静态场景（单帧PNG）
echo "=== 生成静态场景（CUDA） ==="

echo "生成RGB场景..."
echo "y" | ./render_and_convert.sh rgb rgb_scene_cuda 1 cuda $CUDA_OUTPUT_DIR

echo "生成Pattern场景..."
echo "y" | ./render_and_convert.sh pattern pattern_scene_cuda 1 cuda $CUDA_OUTPUT_DIR

echo "生成BigLittle场景..."
echo "y" | ./render_and_convert.sh biglittle biglittle_scene_cuda 1 cuda $CUDA_OUTPUT_DIR

echo "生成Rand10k场景..."
echo "y" | ./render_and_convert.sh rand10k rand10k_scene_cuda 1 cuda $CUDA_OUTPUT_DIR

echo "生成Rand100k场景..."
echo "y" | ./render_and_convert.sh rand100k rand100k_scene_cuda 1 cuda $CUDA_OUTPUT_DIR

echo "生成SnowSingle场景..."
echo "y" | ./render_and_convert.sh snowsingle snowsingle_scene_cuda 1 cuda $CUDA_OUTPUT_DIR

# 动画场景（GIF）
echo "=== 生成动画场景（CUDA） ==="

echo "生成Snow动画..."
echo "y" | ./render_and_convert.sh snow snow_animation_cuda 30 cuda $CUDA_OUTPUT_DIR

echo "生成Pattern动画..."
echo "y" | ./render_and_convert.sh pattern pattern_animation_cuda 20 cuda $CUDA_OUTPUT_DIR

echo "生成BigLittle动画..."
echo "y" | ./render_and_convert.sh biglittle biglittle_animation_cuda 15 cuda $CUDA_OUTPUT_DIR

# 可选：更多动画场景
echo "=== 生成其他动画场景（可选） ==="

echo "生成BouncingBalls动画..."
echo "y" | ./render_and_convert.sh bouncingballs bouncingballs_cuda 25 cuda $CUDA_OUTPUT_DIR

echo "生成Fireworks动画..."
echo "y" | ./render_and_convert.sh fireworks fireworks_cuda 30 cuda $CUDA_OUTPUT_DIR

echo "生成Hypnosis动画..."
echo "y" | ./render_and_convert.sh hypnosis hypnosis_cuda 20 cuda $CUDA_OUTPUT_DIR

echo "所有CUDA场景生成完成！"
echo "查看CUDA输出文件夹："
ls -la $CUDA_OUTPUT_DIR/
