#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>



#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#include <cub/cub.cuh>

#define CIRCLE_CACHE_SIZE 256
#define TILE_WIDTH 25
#define TILE_HEIGHT 25
#define CUDA_CHECK(err) (cuda_check_error(err, __FILE__, __LINE__))

inline void cuda_check_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct CircleData {
    float3 position;
    float  radius;
    float3 color;
};


// tile-circle pair
struct TileCirclePair {
    int tile_id;
    int circle_id;
    
    __host__ __device__
    bool operator<(const TileCirclePair& other) const {
        if (tile_id != other.tile_id) {
            return tile_id < other.tile_id;
        }
        return circle_id < other.circle_id;
    }
};

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;

    TileCirclePair* tileCirclePairs;
    int numTilesX;
    int numTilesY;
    int* tileOffsets;
    int* pairCounter;
    int maxPairs;
    int* cudaDeviceCirclePairCounts;
    int* cudaDeviceCirclePairOffsets;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// count how many pairs each circle will generate
__global__ void kernelCountCirclePairs() {
    int circle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (circle_id >= cuConstRendererParams.numberOfCircles)
        return;

    int index3 = 3 * circle_id;
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[circle_id];

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    int numTilesX = cuConstRendererParams.numTilesX;
    int numTilesY = cuConstRendererParams.numTilesY;

    // bounding box boundaries
    int minX = static_cast<int>(imageWidth * (p.x - rad));
    int maxX = static_cast<int>(imageWidth * (p.x + rad));
    int minY = static_cast<int>(imageHeight * (p.y - rad));
    int maxY = static_cast<int>(imageHeight * (p.y + rad));

    int bboxMinTileX = max(0, minX / TILE_WIDTH);
    int bboxMaxTileX = min(numTilesX - 1, maxX / TILE_WIDTH);
    int bboxMinTileY = max(0, minY / TILE_HEIGHT);
    int bboxMaxTileY = min(numTilesY - 1, maxY / TILE_HEIGHT);

    // center tile
    int centerPixelX = static_cast<int>(p.x * imageWidth);
    int centerPixelY = static_cast<int>(p.y * imageHeight);
    
    int centerTileX = max(0, min(numTilesX - 1, centerPixelX / TILE_WIDTH));
    int centerTileY = max(0, min(numTilesY - 1, centerPixelY / TILE_HEIGHT));

    // always include at least one tile
    int startTileX = min(bboxMinTileX, centerTileX);
    int endTileX   = max(bboxMaxTileX, centerTileX);
    int startTileY = min(bboxMinTileY, centerTileY);
    int endTileY   = max(bboxMaxTileY, centerTileY);

    // Count pairs for this circle
    int pairCount = (endTileX - startTileX + 1) * (endTileY - startTileY + 1);
    cuConstRendererParams.cudaDeviceCirclePairCounts[circle_id] = pairCount;
}

// generate pairs using pre-computed offsets
__global__ void kernelAssignCirclesToTiles() {
    int circle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (circle_id >= cuConstRendererParams.numberOfCircles)
        return;

    int index3 = 3 * circle_id;
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[circle_id];

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    int numTilesX = cuConstRendererParams.numTilesX;
    int numTilesY = cuConstRendererParams.numTilesY;

    // bounding box boundaries
    int minX = static_cast<int>(imageWidth * (p.x - rad));
    int maxX = static_cast<int>(imageWidth * (p.x + rad));
    int minY = static_cast<int>(imageHeight * (p.y - rad));
    int maxY = static_cast<int>(imageHeight * (p.y + rad));

    int bboxMinTileX = max(0, minX / TILE_WIDTH);
    int bboxMaxTileX = min(numTilesX - 1, maxX / TILE_WIDTH);
    int bboxMinTileY = max(0, minY / TILE_HEIGHT);
    int bboxMaxTileY = min(numTilesY - 1, maxY / TILE_HEIGHT);

    // center tile
    int centerPixelX = static_cast<int>(p.x * imageWidth);
    int centerPixelY = static_cast<int>(p.y * imageHeight);
    
    int centerTileX = max(0, min(numTilesX - 1, centerPixelX / TILE_WIDTH));
    int centerTileY = max(0, min(numTilesY - 1, centerPixelY / TILE_HEIGHT));

    // always include at least one tile
    int startTileX = min(bboxMinTileX, centerTileX);
    int endTileX   = max(bboxMaxTileX, centerTileX);
    int startTileY = min(bboxMinTileY, centerTileY);
    int endTileY   = max(bboxMaxTileY, centerTileY);

    int baseOffset = cuConstRendererParams.cudaDeviceCirclePairOffsets[circle_id];
    
    // iterate over the merged safe range
    int localIndex = 0;
    for (int tile_y = startTileY; tile_y <= endTileY; tile_y++) {
        for (int tile_x = startTileX; tile_x <= endTileX; tile_x++) {
            int pair_index = baseOffset + localIndex;
            int tile_id = tile_y * numTilesX + tile_x;
            
            if (pair_index < cuConstRendererParams.maxPairs) {
                cuConstRendererParams.tileCirclePairs[pair_index].tile_id = tile_id;
                cuConstRendererParams.tileCirclePairs[pair_index].circle_id = circle_id;
            }
            localIndex++;
        }
    }
}

// how do I know where my tile, circle pairs start in the pairs?
__global__ void buildTileOffsetsKernel(const TileCirclePair* d_pairs, int* d_offsets, int numPairs, int numTiles) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPairs) {
        return;
    }

    int current_tile_id = d_pairs[idx].tile_id;
    
    if (idx == 0) {
        for (int tile = 0; tile <= current_tile_id; ++tile) {
            d_offsets[tile] = 0;
        }
    } else {
        int previous_tile_id = d_pairs[idx - 1].tile_id;
        
        if (current_tile_id > previous_tile_id) {
            for (int tile = previous_tile_id + 1; tile <= current_tile_id; ++tile) {
                d_offsets[tile] = idx;
            }
        }
    }
    if (idx == numPairs - 1) {
        for (int tile = current_tile_id + 1; tile <= numTiles; ++tile) {
            d_offsets[tile] = numPairs;
        }
    }
}

__host__ void buildTileOffsetsGPU(TileCirclePair* d_pairs, int* d_offsets, int numPairs, int numTiles) {

    if (numPairs == 0) {
        cudaMemset(d_offsets, 0, sizeof(int) * (numTiles + 1));
        return;
    }

    int blockSize = 256;
    int gridSize = (numPairs + blockSize - 1) / blockSize;

    buildTileOffsetsKernel<<<gridSize, blockSize>>>(d_pairs, d_offsets, numPairs, numTiles);
}

__global__ void kernelRenderTiles() {
    __shared__ CircleData circle_cache[CIRCLE_CACHE_SIZE];

    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tile_id = tileY * cuConstRendererParams.numTilesX + tileX;

    int threadX_in_tile = threadIdx.x;
    int threadY_in_tile = threadIdx.y;
    int thread_id_in_block = threadY_in_tile * blockDim.x + threadX_in_tile;

    int pixelX = tileX * TILE_WIDTH + threadX_in_tile;
    int pixelY = tileY * TILE_HEIGHT + threadY_in_tile;
    
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    if (pixelX >= imageWidth || pixelY >= imageHeight) {
        return;
    }

    int offset = 4 * (pixelY * imageWidth + pixelX);
    float4* imagePtr_global = (float4*)(&cuConstRendererParams.imageData[offset]);
    float4 finalColor = *imagePtr_global;

    int startOffset = cuConstRendererParams.tileOffsets[tile_id];
    int endOffset = cuConstRendererParams.tileOffsets[tile_id + 1];
    int numCirclesInTile = endOffset - startOffset;

    if (numCirclesInTile == 0) {
        return;
    }

    for (int page_start = 0; page_start < numCirclesInTile; page_start += CIRCLE_CACHE_SIZE) {
        
        // D1. 协同加载当前页的数据到共享内存
        // 只有块内的前 CIRCLE_CACHE_SIZE 个线程参与加载
        if (thread_id_in_block < CIRCLE_CACHE_SIZE) {
            int circle_index_in_page = thread_id_in_block;
            int circle_index_in_tile = page_start + circle_index_in_page;

            // 确保不会读取超过 `numCirclesInTile` 的范围（处理最后一页）
            if (circle_index_in_tile < numCirclesInTile) {
                int circle_id = cuConstRendererParams.tileCirclePairs[startOffset + circle_index_in_tile].circle_id;
            
                // 从全局内存读取并写入共享内存
                circle_cache[circle_index_in_page].position = *(float3*)(&cuConstRendererParams.position[circle_id * 3]);
                circle_cache[circle_index_in_page].radius   = cuConstRendererParams.radius[circle_id];
            
                if (cuConstRendererParams.sceneName != SNOWFLAKES && cuConstRendererParams.sceneName != SNOWFLAKES_SINGLE_FRAME) {
                     circle_cache[circle_index_in_page].color = *(float3*)&(cuConstRendererParams.color[circle_id * 3]);
                }
            }
        }
        
        // 同步栅栏：确保本页数据全部加载完成，才能进入计算阶段
        __syncthreads();

        // D2. 处理已缓存到共享内存中的数据
        // 计算当前页实际有效的圆数量
        int num_circles_in_this_page = min(CIRCLE_CACHE_SIZE, numCirclesInTile - page_start);
        float pixelCenterX = static_cast<float>(pixelX) + 0.5f;
        float pixelCenterY = static_cast<float>(pixelY) + 0.5f;

        // 内层循环：所有线程（0-1023）处理共享内存中的数据
        for (int i = 0; i < num_circles_in_this_page; i++) {
            CircleData current_circle = circle_cache[i]; // 从快速的共享内存读取
            
            // --- 渲染与颜色混合逻辑 (与您的原始代码相同) ---
            float circleCenterX_pixels = current_circle.position.x * imageWidth;
            float circleCenterY_pixels = current_circle.position.y * imageHeight;
            float radius_pixels = current_circle.radius * imageWidth;

            float diffX_pixels = circleCenterX_pixels - pixelCenterX;
            float diffY_pixels = circleCenterY_pixels - pixelCenterY;
            float distSq_pixels = diffX_pixels * diffX_pixels + diffY_pixels * diffY_pixels;
            float radSq_pixels = radius_pixels * radius_pixels;

            if (distSq_pixels > radSq_pixels) {
                continue;
            }
            float3 rgb;
            float alpha;

            if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;

                float normPixelDist = sqrtf(distSq_pixels) / radius_pixels;
                rgb = lookupColor(normPixelDist);

                float maxAlpha = .6f + .4f * (1.f - current_circle.position.z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
            } else {
                rgb = current_circle.color;
                alpha = .5f;
            }
            
            float oneMinusAlpha = 1.f - alpha;
            finalColor.x = alpha * rgb.x + oneMinusAlpha * finalColor.x;
            finalColor.y = alpha * rgb.y + oneMinusAlpha * finalColor.y;
            finalColor.z = alpha * rgb.z + oneMinusAlpha * finalColor.z;
        }

        // 确保所有线程都完成了本页的计算，在进入下一页加载前同步
        // 对于当前颜色混合算法，此同步点不是严格必需的，但保留它是更安全的设计
        __syncthreads();
    }

    // --------------------------------------------------------------------------
    // E. 将最终结果写回全局内存
    // --------------------------------------------------------------------------
    *imagePtr_global = finalColor;
}
////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;

    cudaDeviceTileCirclePairs = NULL;
    cudaDeviceSortBuffer = NULL;
    cudaDeviceTileOffsets = NULL;
    cudaDevicePairCounter = NULL;
    cudaDeviceSortTempStorage = NULL;
    sortTempStorageBytes = 0;
    cudaDeviceCirclePairCounts = NULL;
    cudaDeviceCirclePairOffsets = NULL;
    cudaDeviceScanTempStorage = NULL;
    scanTempStorageBytes = 0;
    numTilesX = 0;
    numTilesY = 0;
    maxPairs = 0;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }

    if (cudaDeviceTileCirclePairs) {
        cudaFree(cudaDeviceTileCirclePairs);
        cudaFree(cudaDeviceSortBuffer);
        cudaFree(cudaDeviceTileOffsets);
        cudaFree(cudaDevicePairCounter);
        cudaFree(cudaDeviceSortTempStorage);
        cudaFree(cudaDeviceCirclePairCounts);
        cudaFree(cudaDeviceCirclePairOffsets);
        cudaFree(cudaDeviceScanTempStorage);
    }
}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);
    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    numTilesX = (image->width + TILE_WIDTH - 1) / TILE_WIDTH;
    numTilesY = (image->height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int numTiles = numTilesX * numTilesY;
    maxPairs = numberOfCircles * 2500; // Increased safety margin for hypnosis scene

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
    
    // Memory for the tile-based pipeline
    cudaMalloc(&cudaDeviceTileCirclePairs, sizeof(TileCirclePair) * maxPairs);
    cudaMalloc(&cudaDeviceSortBuffer, sizeof(TileCirclePair) * maxPairs);
    cudaMalloc(&cudaDeviceTileOffsets, sizeof(int) * (numTiles + 1)); // Using the correct size
    cudaMalloc(&cudaDevicePairCounter, sizeof(int));
    
    // Memory for exclusive scan approach
    cudaMalloc(&cudaDeviceCirclePairCounts, sizeof(int) * numberOfCircles);
    cudaMalloc(&cudaDeviceCirclePairOffsets, sizeof(int) * numberOfCircles);
    
    // Determine temporary storage size for CUB ExclusiveSum
    cub::DeviceScan::ExclusiveSum(NULL, scanTempStorageBytes, (int*)NULL, (int*)NULL, numberOfCircles);
    cudaMalloc(&cudaDeviceScanTempStorage, scanTempStorageBytes);
    
    // Note: cudaDeviceSortTempStorage is not allocated here because the CUB code
    // to determine its size is commented out. This is fine for CPU sorting.

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.numTilesX = numTilesX;
    params.numTilesY = numTilesY;
    params.tileCirclePairs = cudaDeviceTileCirclePairs;
    params.tileOffsets = cudaDeviceTileOffsets;
    params.pairCounter = cudaDevicePairCounter;
    params.maxPairs = maxPairs;
    params.cudaDeviceCirclePairCounts = cudaDeviceCirclePairCounts;
    params.cudaDeviceCirclePairOffsets = cudaDeviceCirclePairOffsets;
    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };
    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    printf("Total number of circles: %d\n", numberOfCircles);
    dim3 countGridDim((numberOfCircles + 255) / 256);
    dim3 countBlockDim(256);
    kernelCountCirclePairs<<<countGridDim, countBlockDim>>>();
    // cirle, number of tiles covered
    
    cub::DeviceScan::ExclusiveSum(cudaDeviceScanTempStorage, scanTempStorageBytes, 
                                  cudaDeviceCirclePairCounts, cudaDeviceCirclePairOffsets, numberOfCircles);
    // circle, offset of pairs for this circle

    int lastOffset, lastCount;
    cudaMemcpy(&lastOffset, &cudaDeviceCirclePairOffsets[numberOfCircles - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastCount, &cudaDeviceCirclePairCounts[numberOfCircles - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int numPairs = lastOffset + lastCount;
    
    dim3 assignGridDim((numberOfCircles + 255) / 256);
    dim3 assignBlockDim(256);
    kernelAssignCirclesToTiles<<<assignGridDim, assignBlockDim>>>();
    // circle, tile id, tile id
    
    // GPU sorting using Thrust
    thrust::device_ptr<TileCirclePair> dev_pairs_ptr(cudaDeviceTileCirclePairs);
    thrust::sort(dev_pairs_ptr, dev_pairs_ptr + numPairs);
    // tile id, circle id
    
    // GPU build tile offsets
    int numTiles = numTilesX * numTilesY;
    buildTileOffsetsGPU(cudaDeviceTileCirclePairs, cudaDeviceTileOffsets, numPairs, numTiles);

    dim3 renderGridDim(numTilesX, numTilesY);
    dim3 renderBlockDim(TILE_WIDTH, TILE_HEIGHT);
    kernelRenderTiles<<<renderGridDim, renderBlockDim>>>();

    cudaDeviceSynchronize();
}