#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"

// Forward declaration for CUDA-specific structures
struct TileCirclePair;


class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numberOfCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    TileCirclePair* cudaDeviceTileCirclePairs;
    TileCirclePair* cudaDeviceSortBuffer;
    int* cudaDeviceTileOffsets;
    int* cudaDevicePairCounter;
    void* cudaDeviceSortTempStorage;
    size_t sortTempStorageBytes;
    int* cudaDeviceCirclePairCounts;
    int* cudaDeviceCirclePairOffsets;

    int* cudaDeviceTileCounts;
    int* cudaDeviceBucketCounters;
    int* cudaDeviceTotalPairCounter;

    int numTilesX;
    int numTilesY;
    int maxPairs;
    void* cudaDeviceScanTempStorage;
    size_t scanTempStorageBytes;

    int tileWidth;
    int tileHeight;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    // Dynamic tile size configuration interface
    void setTileSize(int width, int height);
    void getTileSize(int& width, int& height) const;

    void shadePixel(
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData, 
        int circleIndex);
};


#endif
