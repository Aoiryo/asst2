#!/bin/bash

# 脚本：render_and_convert.sh
# 功能：运行render程序生成PPM图像序列，然后转换为PNG或GIF动画
# 用法：./render_and_convert.sh <scene_name> [output_name] [frame_count_or_frame_num] [renderer] [output_dir] [specific_frame]
# 示例：
#   生成动画: ./render_and_convert.sh snow my_snow_image 30 cuda cuda_output
#   生成特定帧: ./render_and_convert.sh hypnosis hypnosis_frame5 1 cuda cuda_output 5

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <scene_name> [output_name] [frame_count] [renderer] [output_dir] [specific_frame]"
    echo "有效的场景名称: rgb, rgby, rand10k, rand100k, biglittle, littlebig, pattern, bouncingballs, fireworks, hypnosis, snow, snowsingle"
    echo "示例:"
    echo "  生成动画: $0 snow my_snow_image 30 cuda cuda_output"
    echo "  生成特定帧: $0 hypnosis hypnosis_frame5 1 cuda cuda_output 5"
    echo "frame_count: 帧数，默认为1（静态图），大于1时生成GIF动画"
    echo "renderer: 渲染器，ref（参考实现）或cuda，默认为ref"
    echo "output_dir: 输出目录，默认为output"
    echo "specific_frame: 指定特定帧号（可选），仅在frame_count=1时有效"
    exit 1
fi

SCENE_NAME=$1
OUTPUT_NAME=${2:-$SCENE_NAME}  # 如果没有提供输出名称，使用场景名称
FRAME_COUNT=${3:-1}  # 默认1帧（静态图）
RENDERER=${4:-ref}  # 默认使用reference渲染器
CUSTOM_OUTPUT_DIR=${5:-}  # 自定义输出目录
SPECIFIC_FRAME=${6:-}  # 指定特定帧号（可选）

echo "正在渲染场景: $SCENE_NAME"
echo "输出文件名: $OUTPUT_NAME"
echo "帧数: $FRAME_COUNT"
echo "渲染器: $RENDERER"
if [ -n "$SPECIFIC_FRAME" ]; then
    echo "指定帧号: $SPECIFIC_FRAME"
fi

# 创建PPM文件夹和输出文件夹
PPM_DIR="pmp_frames"
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
else
    OUTPUT_DIR="output"
fi

echo "创建/清空PPM文件夹: $PPM_DIR"
# 如果PPM文件夹存在，清空它；如果不存在，创建它
if [ -d "$PPM_DIR" ]; then
    rm -rf "$PPM_DIR"/*
    echo "已清空现有PPM文件夹"
else
    mkdir -p "$PPM_DIR"
    echo "已创建PPM文件夹"
fi

echo "创建输出文件夹: $OUTPUT_DIR"
# 创建输出文件夹（如果不存在）
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "已创建输出文件夹"
else
    echo "输出文件夹已存在"
fi

# 检查render可执行文件是否存在
if [ ! -f "./render" ]; then
    echo "错误: 在当前目录中找不到render可执行文件"
    exit 1
fi

# 运行render程序生成PPM文件
echo "运行渲染程序..."
if [ "$FRAME_COUNT" -eq 1 ] && [ -n "$SPECIFIC_FRAME" ]; then
    # 生成指定的特定帧
    FRAME_START=$SPECIFIC_FRAME
    FRAME_END=$((SPECIFIC_FRAME + 1))
    echo "渲染特定帧: $SPECIFIC_FRAME"
    ./render -b $FRAME_START:$FRAME_END -f "${PPM_DIR}/${OUTPUT_NAME}_temp" -r "$RENDERER" "$SCENE_NAME"
    # 构建预期的PPM文件名，需要4位数字填充
    FRAME_PADDED=$(printf "%04d" $SPECIFIC_FRAME)
    PPM_PATTERN="${PPM_DIR}/${OUTPUT_NAME}_temp_${FRAME_PADDED}.ppm"
elif [ "$FRAME_COUNT" -eq 1 ]; then
    # 生成单帧静态图（第0帧）
    ./render -b 0:1 -f "${PPM_DIR}/${OUTPUT_NAME}_temp" -r "$RENDERER" "$SCENE_NAME"
    PPM_PATTERN="${PPM_DIR}/${OUTPUT_NAME}_temp_0000.ppm"
else
    # 生成多帧动画
    ./render -b 0:$FRAME_COUNT -f "${PPM_DIR}/${OUTPUT_NAME}_temp" -r "$RENDERER" "$SCENE_NAME"
    PPM_PATTERN="${PPM_DIR}/${OUTPUT_NAME}_temp_*.ppm"
fi

# 检查render是否成功
if [ $? -ne 0 ]; then
    echo "错误: 渲染失败"
    exit 1
fi

# 检查PPM文件是否生成
if [ "$FRAME_COUNT" -eq 1 ] && [ -n "$SPECIFIC_FRAME" ]; then
    # 检查特定帧的PPM文件
    FRAME_PADDED=$(printf "%04d" $SPECIFIC_FRAME)
    EXPECTED_PPM="${PPM_DIR}/${OUTPUT_NAME}_temp_${FRAME_PADDED}.ppm"
    if [ ! -f "$EXPECTED_PPM" ]; then
        echo "错误: 找不到生成的PPM文件: $EXPECTED_PPM"
        exit 1
    fi
    echo "PPM文件已生成: $EXPECTED_PPM"
else
    # 检查常规PPM文件
    PPM_FILES=(${PPM_DIR}/${OUTPUT_NAME}_temp_*.ppm)
    if [ ! -f "${PPM_FILES[0]}" ]; then
        echo "错误: 找不到生成的PPM文件在目录: $PPM_DIR"
        exit 1
    fi
    echo "PPM文件已生成在目录: $PPM_DIR"
    echo "生成了 ${#PPM_FILES[@]} 个PPM文件"
fi

# 检查是否安装了ImageMagick的convert命令
if command -v convert >/dev/null 2>&1; then
    if [ "$FRAME_COUNT" -eq 1 ]; then
        # 单帧转换为PNG
        echo "使用ImageMagick转换PPM到PNG..."
        if [ -n "$SPECIFIC_FRAME" ]; then
            # 转换特定帧
            FRAME_PADDED=$(printf "%04d" $SPECIFIC_FRAME)
            convert "${PPM_DIR}/${OUTPUT_NAME}_temp_${FRAME_PADDED}.ppm" "${OUTPUT_DIR}/${OUTPUT_NAME}.png"
        else
            # 转换第0帧
            convert "${PPM_DIR}/${OUTPUT_NAME}_temp_0000.ppm" "${OUTPUT_DIR}/${OUTPUT_NAME}.png"
        fi
        CONVERT_SUCCESS=$?
        OUTPUT_TYPE="PNG图片"
        OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.png"
    else
        # 多帧转换为GIF动画
        echo "使用ImageMagick转换PPM序列到GIF动画..."
        convert -delay 10 "${PPM_DIR}/${OUTPUT_NAME}_temp_"*.ppm "${OUTPUT_DIR}/${OUTPUT_NAME}.gif"
        CONVERT_SUCCESS=$?
        OUTPUT_TYPE="GIF动画"
        OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.gif"
    fi
elif command -v magick >/dev/null 2>&1; then
    if [ "$FRAME_COUNT" -eq 1 ]; then
        echo "使用ImageMagick转换PPM到PNG..."
        if [ -n "$SPECIFIC_FRAME" ]; then
            # 转换特定帧
            FRAME_PADDED=$(printf "%04d" $SPECIFIC_FRAME)
            magick "${PPM_DIR}/${OUTPUT_NAME}_temp_${FRAME_PADDED}.ppm" "${OUTPUT_DIR}/${OUTPUT_NAME}.png"
        else
            # 转换第0帧
            magick "${PPM_DIR}/${OUTPUT_NAME}_temp_0000.ppm" "${OUTPUT_DIR}/${OUTPUT_NAME}.png"
        fi
        CONVERT_SUCCESS=$?
        OUTPUT_TYPE="PNG图片"
        OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.png"
    else
        echo "使用ImageMagick转换PPM序列到GIF动画..."
        magick -delay 10 "${PPM_DIR}/${OUTPUT_NAME}_temp_"*.ppm "${OUTPUT_DIR}/${OUTPUT_NAME}.gif"
        CONVERT_SUCCESS=$?
        OUTPUT_TYPE="GIF动画"
        OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.gif"
    fi
elif command -v pnmtopng >/dev/null 2>&1 && [ "$FRAME_COUNT" -eq 1 ]; then
    echo "使用netpbm工具转换PPM到PNG..."
    if [ -n "$SPECIFIC_FRAME" ]; then
        # 转换特定帧
        FRAME_PADDED=$(printf "%04d" $SPECIFIC_FRAME)
        pnmtopng "${PPM_DIR}/${OUTPUT_NAME}_temp_${FRAME_PADDED}.ppm" > "${OUTPUT_DIR}/${OUTPUT_NAME}.png"
    else
        # 转换第0帧
        pnmtopng "${PPM_DIR}/${OUTPUT_NAME}_temp_0000.ppm" > "${OUTPUT_DIR}/${OUTPUT_NAME}.png"
    fi
    CONVERT_SUCCESS=$?
    OUTPUT_TYPE="PNG图片"
    OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.png"
else
    echo "警告: 没有找到合适的图像转换工具"
    if [ "$FRAME_COUNT" -eq 1 ]; then
        echo "对于单帧转换，请安装以下工具之一:"
        echo "  - ImageMagick: sudo apt-get install imagemagick"
        echo "  - netpbm: sudo apt-get install netpbm"
    else
        echo "对于GIF动画转换，请安装:"
        echo "  - ImageMagick: sudo apt-get install imagemagick"
    fi
    echo "PPM文件已保存在目录: $PPM_DIR"
    exit 1
fi

# 检查转换是否成功
if [ $CONVERT_SUCCESS -eq 0 ]; then
    echo "转换成功! ${OUTPUT_TYPE}已保存为: $OUTPUT_FILE"
    
    # 询问是否删除临时PPM文件
    echo -n "是否删除临时PPM文件夹? (y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$PPM_DIR"
        echo "已删除临时PPM文件夹"
    else
        echo "保留PPM文件夹: $PPM_DIR"
    fi
else
    echo "错误: 转换失败"
    exit 1
fi

echo "完成!"
