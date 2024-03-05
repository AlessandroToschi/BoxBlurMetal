# Reverse Engineering Metal Performance Shaders (MPS) Box Blur

Tech: GPU, Metal, macOS
GitHub: https://github.com/AlessandroToschi/BoxBlurMetal
Last Update: March 5, 2024 5:05 PM

I got this idea: let’s try to derive how the **box blur** is implemented in **Metal Performance Shaders (MPS)** and see if I’m able to provide a similar implementation in terms of performance.

**Spoiler: I made it and it is faster*.**

But first, let’s start in order:

- [Box Blur](https://www.notion.so/Reverse-Engineering-Metal-Performance-Shaders-MPS-Box-Blur-94ddfe603f19468e923444165b011991?pvs=21): quick recall of what box blur is, how it works, and the difference between single and double passes.
- [MPS Box Blur](https://www.notion.so/Reverse-Engineering-Metal-Performance-Shaders-MPS-Box-Blur-94ddfe603f19468e923444165b011991?pvs=21): reverse engineering of the MPS implementation through an in-depth analysis using the Xcode GPU capture tool.
- [Implementations](https://www.notion.so/Reverse-Engineering-Metal-Performance-Shaders-MPS-Box-Blur-94ddfe603f19468e923444165b011991?pvs=21): from the slowest and naive single pass implementation to a more sophisticated and performant double pass.
- [Performance](https://www.notion.so/Reverse-Engineering-Metal-Performance-Shaders-MPS-Box-Blur-94ddfe603f19468e923444165b011991?pvs=21): let’s compare the different implementations and check which is faster.
- [Do’s and Don’ts](https://www.notion.so/Reverse-Engineering-Metal-Performance-Shaders-MPS-Box-Blur-94ddfe603f19468e923444165b011991?pvs=21): tips and tricks I learned on GPGPU and Metal.

# BoxBlur

The Box Blur is a **neighbor operation** that, for every pixel, computes the **average of the pixel itself and its neighbors**. The ***radius*** of the box blur is a positive integer that represents how many pixels in every direction are averaged.

$$
BoxBlur(x, y) = \frac{\sum_{i=-r}^{r}\sum_{j=-r}^{r}image(x+i, y+j)}{(2 r +1)^2}
$$

The complexity of this operator is $O(N*(2r+1)^2)$ where $N$ is the number of pixels and $r$ is the radius. If implemented following the formula above, it is known as a ***single pass box blur***.

- Single Pass Pseudocode
    
    ```swift
    let inputImage = ...
    let outputImage = ...
    let radius = ...
    
    for y in 0 ..< inputImage.height {
    	for x in 0 ..< inputImage.width {
    		sum = 0
    		for j in -radius ... radius {
    			for i in -radius ... radius {
    				sum += inputImage[x + i, y + j]
    			}
    		}
    		outputImage[x, y] = sum / pow((2 * radius + 1), 2)
    	}
    }
    ```
    

Luckily, the box blur is a [separable filter](https://en.wikipedia.org/wiki/Separable_filter), meaning that we can decompose the operation from a single 2D pass to **two 1D passes (*double pass box blur*)**, one horizontal and one vertical. This lowers the complexity to $O(N(2r+1))$, but at the additional cost of storing the intermediate results of the first horizontal pass.

- Double Pass Pseudocode
    
    ```swift
    let inputImage = ...
    let outputImage = ...
    let tempImage = ...
    let radius = ...
    
    for y in 0 ..< inputImage.height {
    	for x in 0 ..< inputImage.width {
    		let sum = 0
    		for i in -radius ... radius {
    			sum += inputImage[x + i, y]
    		}
    		tempImage[x, y] = sum / (2 * radius + 1)
    	}
    }
    
    for y in 0 ..< tempImage.height {
    	for x in 0 ..< tempImage.width {
    		let sum = 0
    		for j in -radius ... radius {
    			sum += tempImage[x, y + j]
    		}
    		outputImage[x, y] = sum / (2 * radius + 1)
    	}
    }
    ```
    

The double pass can be further optimized through [accumulation](https://web.archive.org/web/20060718054020/http://www.acm.uiuc.edu/siggraph/workshops/wjarosz_convolution_2001.pdf), where the sum of each pixel is reused by adding the next pixel and subtracting the last pixel in the blurring range $[-r; r]$, lowering the complexity to $O(N)$. The biggest advantage of accumulation is that **we’re no longer dependent on the radius, so the execution time is the “same” for every radius**.

- Accumulated Double Pass Pseudocode
    
    ```swift
    let inputImage = ...
    let outputImage = ...
    let tempImage = ...
    let radius = ...
    
    for y in 0 ..< inputImage.height {
    	var sum = 0
    	for i in -radius ..< radius {
    		sum += inputImage[i, y]
    	}
    	for x in 0 ..< inputImage.width {
    		sum = sum + inputImage[x + radius, y] - inputImage[x - radius, y]
    		tempImage[x, y] = sum / (2 * radius + 1)
    	}
    }
    
    for x in 0 ..< tempImage.width {
    	var sum = 0
    	for j in -radius ..< radius {
    		sum += inputImage[x, j]
    	}
    	for y in 0 ..< tempImage.height {
    		sum = sum + tempImage[x, y + radius] - tempImage[x, y - radius]
    		outputImage[x, y] = sum / (2 * radius + 1)
    	}
    }
    ```
    

# MPS Box Blur

The only way we have to infer how the box blur is implemented in MPS is to utilize the Xcode GPU capture tool and make educated guesses based on the GPU trace. I set up a small script to capture the GPU workload, and here are some insights and details:

- [GPU Trace](https://www.icloud.com/iclouddrive/0f0HUdIh8nfVM-a_AFn292fcA#MPSTrace)
- Input: **BGRA8Unorm** 3024x4032
- Output: same as input
- Radius: 30 —> Kernel size = 61x61
- It is implemented using a double pass technique: `MIBox_Horizontal` first and then `MIBox_Vertical_Scan` after.

<img width="459" alt="Screenshot_2024-03-02_at_12 14 22" src="https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/da902138-f14e-42d2-9d84-9b1519ab09ce">

- The intermediate buffer is the same size as the input but the pixel format is **RGBA32Float**.
    
    <img width="646" alt="Screenshot_2024-03-01_at_16 04 52" src="https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/28547682-4f90-4b35-a197-9f09ef9bb55e">

- On average, the `MIBox_Horizontal` kernel takes **1.5ms** per run, while the `MIBox_Vertical_Scan` kernel takes between **2ms and 2.2ms.**

<img width="616" alt="Untitled" src="https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/ec83793f-fc3e-4368-a373-6eb4b62a52b7">

MIBox_Horizontal

<img width="610" alt="222_ms" src="https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/5e87057d-5bad-4941-94ff-ab8e13e1fc8d">

MIBox_Vertical_Scan

## MIBox_Horizontal

- The dispatch grid is `{126, 1, 1} x {32, 1, 1}` :
    - 126 x 32 = 4032 (image height) —> **each thread processes one row of the image.**
    - 32 is the simdgroup execution width.
    - Each thread samples ~ 6020 pixels, roughly twice the width of the image —> this suggests that the kernel is performing **accumulation** where each pixel is accessed twice. Otherwise we should see a higher number of pixels per kernel, close to `width x (2 * radius + 1)`.
        
        <img width="629" alt="Kernel_Occupancy" src="https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/ae32ba4f-ba41-459c-bffb-d29c136a64b3">

- The kernel reads ~ 20Gib/s and writes ~ 140-150 Gib/s.
    - The kernel is limited by the texture write, reaching 97-98%.
    - The write is also amplified by the fact we’re reading RGBA8 and writing RGBA32, this is 4x amplification factor.

## MIBox_Vertical_Scan

- The dispatch grid is `{189, 1, 1} x {16, 16, 1}` :
    - 189 x 16 = 3024 (image width)
    - Differently from before, **each column is processed by 16 threads cooperatively**.
    - Each thread samples 506 pixels —> ~ 2x image height ( 506 pixels x 16 threads ~ 2 x image height).
        
        <img width="622" alt="Kernel_ALU_Active_Time" src="https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/3e16b45f-14ab-41c6-a999-a101a880897c">

    - Another difference from before is that the vertical pass likely employs the prefix scan algorithm because it’s indicated in the kernel name, and the kernel utilizes threadgroup memory, commonly used in this type of algorithm.
    - Threadgroup memory size: 8704 bytes —> 544 bytes per column —> 34 bytes per thread.
    - This kernel has a better occupancy (~ 50%) and it is mainly limited by ALU.
    - It's unclear why a different implementation is used for the vertical pass compared to the horizontal one. I will explore the prefix sum algorithm in another post.

# Implementations

### Single Pass

The single pass is the simplest implementation possible in which each thread processes one pixel.

```cpp
[[ kernel ]]
void box_blur_single_pass(
	texture2d<float, access::sample> input_texture [[ texture(0) ]],
  texture2d<float, access::read_write> output_texture [[ texture(1) ]],
  constant int& radius [[ buffer(0) ]],
  constant int& kernel_size [[ buffer(1) ]],
  uint2 xy [[ thread_position_in_grid ]]
) {
  float4 sum = 0.0f;
  const float divider = kernel_size * kernel_size;

	constexpr sampler blur_sampler = sampler(
		coord::pixel, 
		address::clamp_to_edge, 
		filter::nearest
	);

  for(int offsetY = -radius; offsetY <= radius; offsetY++) {
    for(int offsetX = -radius; offsetX <= radius; offsetX++) {
      sum += input_texture.sample(
				blur_sampler, 
				float2(xy.x + offsetX, xy.y + offsetY)
			);
    }
  }

  output_texture.write(sum / divider, xy);
}
```

Pros:

- No temporary memory required.

Cons:

- No pixel reuse among iterations (the same pixel is sampled multiple times).
- The execution time depends on the radius.
- Each thread performs minimal work since it’s heavily memory bound.

Improvements:

- Process more pixels per threads, up to one row per thread.
- Reduce the number of pixel read by leveraging the linear filter of the sampler:
    - Sampling the image at `(x + 0.5, y + 0.5)` will sample and mix 4 pixels at once: `(x, y), (x + 1, y + 1), (x + 1, y), (x, y + 1)` , thus reducing the number of samples from `(2r + 1) x (2r + 1)` to `(r + 1) x (r + 1)`.
    - Metal Single Pass Blur With Linear Sampling
        
        ```cpp
        [[ kernel ]]
        void box_blur_single_pass_linear(
        	texture2d<float, access::sample> input_texture [[ texture(0) ]],
          texture2d<float, access::read_write> output_texture [[ texture(1) ]],
          constant int& radius [[ buffer(0) ]],
          constant int& kernel_size [[ buffer(1) ]],
          uint2 xy [[ thread_position_in_grid ]]) 
        {
          constexpr sampler linear_blur_sampler = sampler(
        		coord::pixel, 
        		address::clamp_to_edge, 
        		filter::linear
        	);
          
          float4 sum = 0.0f;
          int count = 0;
          
          float2 uv = float2(xy) - (float)radius + 0.5f;
          const float2 end = float2(xy) + (float)radius - 1.0f;
          
          while (uv.y < end.y) {
            while (uv.x < end.x) {
              sum += input_texture.sample(linear_blur_sampler, uv);
              count++;
        
              uv.x += 2.0f;
            }
            
            uv.x -= 0.5f;
            
            sum += input_texture.sample(linear_blur_sampler, uv);
            count++;
            
            uv.x = xy.x - radius + 0.5f;
            uv.y += 2.0f;
          }
          
          uv.y -= 0.5f;
          
          while (uv.x < end.x) {
            sum += input_texture.sample(linear_blur_sampler, uv);
            count++;
        
            uv.x += 2.0f;
          }
          
          uv.x -= 0.5f;
          
          sum += input_texture.sample(linear_blur_sampler, uv);
          count++;
          
          output_texture.write(sum / count, xy);
        }
        ```
        

### Double Pass

In the double pass, each thread is processing one pixel for each pass, horizontal first and then vertical. We need an intermediate buffer to store the horizontal pass output.

```cpp
[[ kernel ]]
void box_blur_double_pass_h(
    texture2d<float, access::sample> input_texture [[ texture(0) ]],
    texture2d<float, access::read_write> output_texture [[ texture(1) ]],
    constant int& radius [[ buffer(0) ]],
    constant int& kernel_size [[ buffer(1) ]],
    uint2 xy [[ thread_position_in_grid ]]
  ) {
  float4 sum = 0.0f;

  constexpr sampler blur_sampler = sampler(
    coord::pixel, 
    address::clamp_to_edge, 
    filter::nearest
  );

  for(int offsetX = -radius; offsetX <= radius; offsetX++) {
    sum += input_texture.sample(
      blur_sampler, 
      float2(xy.x + offsetX, xy.y)
    );
  }
  output_texture.write(sum / kernel_size, xy);
}

[[ kernel ]]
void box_blur_double_pass_v(
    texture2d<float, access::sample> input_texture [[ texture(0) ]],
    texture2d<float, access::read_write> output_texture [[ texture(1) ]],
    constant int& radius [[ buffer(0) ]],
    constant int& kernel_size [[ buffer(1) ]],
    uint2 xy [[ thread_position_in_grid ]]
  ) {
  float4 sum = 0.0f;

  constexpr sampler blur_sampler = sampler(
    coord::pixel, 
    address::clamp_to_edge, 
    filter::nearest
  );

  for(int offsetY = -radius; offsetY <= radius; offsetY++) {
    sum += input_texture.sample(
      blur_sampler, 
      float2(xy.x, xy.y + offsetY)
    );
  }
  output_texture.write(sum / kernel_size, xy);
}
```

The Pros, Cons, and Improvements are quite the same as single blur but this time we’re reducing the number of operations but at the expense of an intermediate texture.

- Metal Double Pass Blur With Linear Sampling
    
    ```cpp
    [[ kernel ]]
    void box_blur_double_pass_h_linear(
      texture2d<float, access::sample> input_texture [[ texture(0) ]],
        texture2d<float, access::read_write> output_texture [[ texture(1) ]],
        constant int& radius [[ buffer(0) ]],
        constant int& kernel_size [[ buffer(1) ]],
        uint2 xy [[ thread_position_in_grid ]]
      ) {
      float4 sum = 0.0f;
      
      constexpr sampler linear_blur_sampler = sampler(
        coord::pixel, 
        address::clamp_to_edge, 
        filter::linear
      );
      
      float2 uv = float2(xy) - radius + 0.5f;
      const float end = xy.x + radius - 1.0f;
      int count = 0;
      
      while (uv.y < end) {
        sum += input_texture.sample(blur_sampler, uv);
        count++;
        
        uv.x += 2.0f;
      }
    
      uv.x -= 0.5f;
      
      sum += input_texture.sample(blur_sampler, uv);
      count++;
      
      output_texture.write(sum / count, xy);
    }
    
    [[ kernel ]]
    void box_blur_double_pass_v_linear(
        texture2d<float, access::sample> input_texture [[ texture(0) ]],
        texture2d<float, access::read_write> output_texture [[ texture(1) ]],
        constant int& radius [[ buffer(0) ]],
        constant int& kernel_size [[ buffer(1) ]],
        uint2 xy [[ thread_position_in_grid ]]
      ) {
      float4 sum = 0.0f;
      
      constexpr sampler linear_blur_sampler = sampler(
        coord::pixel, 
        address::clamp_to_edge, 
        filter::linear
      );
      
      float2 uv = float2(xy) - radius + 0.5f;
      const float end = xy.y + radius - 1.0f;
      int count = 0;
      
      while (uv.y < end) {
        sum += input_texture.sample(blur_sampler, uv);
        count++;
        
        uv.y += 2.0f;
      }
    
      uv.y -= 0.5f;
      
      sum += input_texture.sample(blur_sampler, uv);
      count++;
      
      output_texture.write(sum / count, xy);
    }
    ```
    

### Double Pass + Accumulation + Loop Unrolling

1. Increase the thread workload: similar to MPS, reduce the number of threads while increasing the work per thread by processing the full row/column.
2. Reuse the pixels through accumulation: every pixel will be sampled twice, one when entering in the blurring range, and one when exiting the blurring range. This will make the implementation independent from the radius.
3. Loop unrolling: process more pixels at each iteration to coalesce reads and writes to texture units.
4. Intermediate texture: choose the appropriate pixel format for the intermediate buffer. MPS uses `RGBA32Float`, which offers high precision but at the expense of bandwidth. For reasonably sized radius and non-critical precision, consider smaller pixel formats such as `RGBA8Unorm` or `RGBA16Float` to reduce memory and bandwidth usage.

```cpp
constexpr sampler blur_sampler = sampler(
	coord::pixel, 
	address::clamp_to_edge, 
	filter::nearest
);

template <
	int N, 
  typename T, 
  typename = typename enable_if<is_same<T, float>::value || is_same<T, half>::value>::type
>
METAL_FUNC void hbox_blur(
    texture2d<T, access::sample> input_texture,
    texture2d<T, access::read_write> output_texture,
    constant int& radius,
    constant int& kernel_size,
    uint row
) {
  if (row >= input_texture.get_height()) {
    return;
  }
  
  vec<T, 4> output[N]; // Sum
  // Pixels exiting the blurring range, to be subtracted to the output.
  vec<T, 4> previousInput[N]; 
  // Pixels entering the blurring range, to be added to the output.
  vec<T, 4> nextInput[N];
  
  // Fill the first sum up to the radius.
  vec<T, 4> start = input_texture.sample(blur_sampler, float2(0.0f, row));
  output[0] = start * (radius + (T)1.0);
  
  for (int i = 1; i < radius; i++) {
    output[0] += input_texture.sample(blur_sampler, float2(i, row));
  }
  
  const int width = static_cast<int>(input_texture.get_width());
  
  previousInput[0] = vec<T, 4>((T)0.0);
  for (int i = 1; i < N; i++) {
    previousInput[i] = input_texture.sample(blur_sampler, float2(-radius + i - 1, row));
  }
  
  for (int i = 0; i < N; i++) {
    nextInput[i] = input_texture.sample(blur_sampler, float2(radius + i, row));
  }
  
  for (int column = 0; column < (width / N); column++) {
    const int c = column * N;
    const int previousIndex = c + N - radius - 1;
    const int nextIndex = c + N + radius;
    
    output[0] = output[0] - previousInput[0] + nextInput[0];
    for (int i = 1; i < N; i++) {
      output[i] = output[i - 1] - previousInput[i] + nextInput[i];
    }
    
    for (int i = 0; i < N; i++) {
      output_texture.write(output[i] / kernel_size, uint2(c + i, row));
    }
    
    for (int i = 0; i < N; i++) {
      previousInput[i] = input_texture.sample(blur_sampler, float2(previousIndex + i, row));
    }
    
    for (int i = 0; i < N; i++) {
      nextInput[i] = input_texture.sample(blur_sampler, float2(nextIndex + i, row));
    }
    
    output[0] = output[N - 1];
  }
  
  const int remainder = width % N;
  
  for (int c = width - remainder; c < width; c++) {
    const int previousIndex = c + N - radius - 1;
    const int nextIndex = c + N + radius;
    
    output[0] = output[0] - previousInput[0] + nextInput[0];
    output_texture.write(output[0] / kernel_size, uint2(c, row));
    previousInput[0] = input_texture.sample(blur_sampler, float2(previousIndex, row));
    nextInput[0] = input_texture.sample(blur_sampler, float2(nextIndex, row));
  }
}

template <
	int N, 
	typename T, 
	typename = typename enable_if<is_same<T, float>::value || is_same<T, half>::value>::type
>
METAL_FUNC void vbox_blur(
    texture2d<T, access::sample> input_texture,
    texture2d<T, access::read_write> output_texture,
    constant int& radius,
    constant int& kernel_size,
    uint column
) {
  if (column >= input_texture.get_width()) {
    return;
  }
  
  vec<T, 4> output[N];
  vec<T, 4> previousInput[N];
  vec<T, 4> nextInput[N];
  
  vec<T, 4> start = input_texture.sample(blur_sampler, float2(column, 0.0f));
  output[0] = start * (radius + (T)1.0);
  
  for (int i = 1; i < radius; i++) {
    output[0] += input_texture.sample(blur_sampler, float2(column, i));
  }
  
  const int height = static_cast<int>(input_texture.get_height());
  
  previousInput[0] = vec<T, 4>((T)0.0);
  for (int i = 1; i < N; i++) {
    previousInput[i] = input_texture.sample(blur_sampler, float2(column, -radius + i - 1));
  }
  
  for (int i = 0; i < N; i++) {
    nextInput[i] = input_texture.sample(blur_sampler, float2(column, radius + i));
  }
  
  for (int row = 0; row < height / N; row++) {
    const int r = row * N;
    const int previousIndex = r + N - radius - 1;
    const int nextIndex = r + N + radius;
    
    for (int i = 0; i < N; i++) {
      output[i] = output[max(0, i - 1)] - previousInput[i] + nextInput[i];
    }
    
    for (int i = 0; i < N; i++) {
      output_texture.write(output[i] / kernel_size, uint2(column, r + i));
    }
    
    for (int i = 0; i < N; i++) {
      previousInput[i] = input_texture.sample(blur_sampler, float2(column, previousIndex + i));
    }
    
    for (int i = 0; i < N; i++) {
      nextInput[i] = input_texture.sample(blur_sampler, float2(column, nextIndex + i));
    }
    
    output[0] = output[N - 1];
  }
  
  const int remainder = height % N;
  
  for (int r = height - remainder; r < height; r++) {
    const int previousIndex = r + N - radius - 1;
    const int nextIndex = r + N + radius;
    
    output[0] = output[0] - previousInput[0] + nextInput[0];
    output_texture.write(output[0] / kernel_size, uint2(column, r));
    previousInput[0] = input_texture.sample(blur_sampler, float2(column, previousIndex));
    nextInput[0] = input_texture.sample(blur_sampler, float2(column, nextIndex));
  }
}
```

# Performance

I created a test set to profile the execution of any possible implementations for different radius and intermediate pixel format.

- Pixel formats: `RGBA8`, `RGBA16Float`, `RGBA32Float`.
- Radii: 1 to 63 (3x3 to 127x127).
- For each pixel format and radius, 10 runs.
- Implementations:
    - Single pass
    - Single pass with linear sampling
    - Double pass
    - Double pass with linear sampling
    - Optimized double pass:
        - 1, 2, 4, 8, 12, 16, 24, 32 elements per iteration as part of the loop unrolling.
    - MPSImageBox
- In total: 63 (radii) x 3 (pixel formats) x 13 (implementations) x 10 (runs) = 24570 executions.
- Raw data:
    [stats.csv](https://www.icloud.com/iclouddrive/0dactCXIRttAeWBCYd1iMpnCg#stats)
    [Google SpreadSheet](https://docs.google.com/spreadsheets/d/13RI71toeBmf2Hol8ubxYyucMxnx5iyTwCyG5GBLpngc/edit?usp=sharing)
    

### Results

- Single pass and double pass are slow:
    - Both execution times are tied to the radius.
    - Single pass takes more than 1,5 seconds for larger radii (> 58).
      ![chart](https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/60e4f4ea-3197-4d62-a7a5-307010fc1959)
        
    - The linear sampling versions are faster:
        - Linear sampling for the single pass can speedup more than **4x**.
        - Linear sampling for the double pass can speedup up to **2x**.
    - The double pass seems not to be heavily penalized by the intermediate texture pixel format, slowing down of a few milliseconds when using `RGBA32Float`.
- The `MPSImageBox` runs stable in 3.1 - 3.4 ms, internally, it uses an intermediate buffer of format `RGBA32Float` .
- The fast double pass (loop unrolling + accumulation) is faster than the previous implementations and, for some pixels per items (1, 2, 4, 8), **is even faster than MPSImageBox**.

![chart_(3)](https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/810b6725-f86e-491f-a73c-f17f2e7d5f66)

RGBA8 intermediate texture pixel format.

- Fast double pass implementations processing more than 8 pixels per iteration are slower than the other, worth trying anyway.
- `fast_double_pass_x8` is the winner in every run:
    - 1.3ms using RGBA8 intermediate texture.
    - 1.8ms using RGBA16 intermediate texture.
    - 3ms using RGBA32 intermediate texture.
- In any case, `fast_double_pass_x2`, `fast_double_pass_x4`, `fast_double_pass_x8` are faster than MPS, even when using `RGBA32` as pixel format of the intermediate texture.
    
    ![chart_(4)](https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/11890052-a54f-4cbc-a153-351dfb783866)

    RGBA8 intermediate texture pixel format.
    
    In the worst case, when using `RGBA32`, **we’re performing close to MPS, which was the original intent 🎉**
    
    ![chart_(5)](https://github.com/AlessandroToschi/BoxBlurMetal/assets/6044244/c7439d67-bbe4-492a-ae34-ebb5f38dfad7)


# Do’s and Don’ts

- Let the sampler sample:
    - You can halve the number of texture sample required by using a linear filter instead of the nearest pixel.
    - The sampler gives you for free different way to handle edges: clamp to edge, mirror, or clamp to zero. There is no need to handle it manually in the kernel.
- Less threads with more workload are better and faster then a lot of threads with very small workload for this algorithm.
- Sampling textures by columns or rows is practically the same, with little or no penalty involved. This is due to tiled memory of Apple Silicon GPUs, might not be the same on other architectures and for sure for linear device memory (buffer).
- Threadgroup memory is slower than sampling the texture for this algorithm.
- Coalescing reads and writes using loop unrolling really improves performance.
