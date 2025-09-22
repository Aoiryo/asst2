#!/bin/bash

# 批量生成所有场景的脚本
# 用法: ./generate_all_scenes.sh

echo "开始批量生成所有场景..."

# 静态场景（单帧PNG）
echo "=== 生成静态场景 ==="

echo "生成RGB场景..."
echo "y" | ./render_and_convert.sh rgb rgb_scene 1

echo "生成Pattern场景..."
echo "y" | ./render_and_convert.sh pattern pattern_scene 1

echo "生成BigLittle场景..."
echo "y" | ./render_and_convert.sh biglittle biglittle_scene 1

echo "生成Rand10k场景..."
echo "y" | ./render_and_convert.sh rand10k rand10k_scene 1

echo "生成Rand100k场景..."
echo "y" | ./render_and_convert.sh rand100k rand100k_scene 1

echo "生成SnowSingle场景..."
echo "y" | ./render_and_convert.sh snowsingle snowsingle_scene 1

# 可选：测试一些场景的动画版本
echo "=== 生成动画场景（可选） ==="

echo "生成Pattern动画..."
echo "y" | ./render_and_convert.sh pattern pattern_animation 20

echo "生成BigLittle动画..."
echo "y" | ./render_and_convert.sh biglittle biglittle_animation 15

echo "所有场景生成完成！"
echo "查看输出文件夹："
ls -la output/
