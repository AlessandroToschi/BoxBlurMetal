//
//  BoxBlur.metal
//  BoxBlur
//
//  Created by Alessandro Toschi on 27/02/24.
//

#include <metal_stdlib>

using namespace metal;

constexpr sampler blur_sampler = sampler(coord::pixel, address::clamp_to_edge, filter::nearest);

template <int N, typename T, typename = typename enable_if<is_same<T, float>::value || is_same<T, half>::value>::type>
METAL_FUNC void hbox_blur(texture2d<T, access::sample> input_texture,
                          texture2d<T, access::read_write> output_texture,
                          constant int& radius,
                          constant int& kernel_size,
                          uint row) {
  if (row >= input_texture.get_height()) {
    return;
  }
  
  vec<T, 4> output[N];
  vec<T, 4> previousInput[N];
  vec<T, 4> nextInput[N];
  
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

template <int N, typename T, typename = typename enable_if<is_same<T, float>::value || is_same<T, half>::value>::type>
METAL_FUNC void vbox_blur(texture2d<T, access::sample> input_texture,
                          texture2d<T, access::read_write> output_texture,
                          constant int& radius,
                          constant int& kernel_size,
                          uint column) {
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

[[ kernel ]]
void box_blur_single_pass(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                          texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                          constant int& radius [[ buffer(0) ]],
                          constant int& kernel_size [[ buffer(1) ]],
                          uint2 xy [[ thread_position_in_grid ]]) {
  float4 sum = 0.0f;
  const float divider = kernel_size * kernel_size;
  for(int offsetY = -radius; offsetY <= radius; offsetY++) {
    for(int offsetX = -radius; offsetX <= radius; offsetX++) {
      sum += input_texture.sample(blur_sampler, float2(xy.x + offsetX, xy.y + offsetY));
    }
  }
  output_texture.write(sum / divider, xy);
}

[[ kernel ]]
void box_blur_double_pass_h(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                            texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                            constant int& radius [[ buffer(0) ]],
                            constant int& kernel_size [[ buffer(1) ]],
                            uint2 xy [[ thread_position_in_grid ]]) {
  float4 sum = 0.0f;
  for(int offsetX = -radius; offsetX <= radius; offsetX++) {
    sum += input_texture.sample(blur_sampler, float2(xy.x + offsetX, xy.y));
  }
  output_texture.write(sum / kernel_size, xy);
}

[[ kernel ]]
void box_blur_double_pass_v(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                            texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                            constant int& radius [[ buffer(0) ]],
                            constant int& kernel_size [[ buffer(1) ]],
                            uint2 xy [[ thread_position_in_grid ]]) {
  float4 sum = 0.0f;
  for(int offsetY = -radius; offsetY <= radius; offsetY++) {
    sum += input_texture.sample(blur_sampler, float2(xy.x, xy.y + offsetY));
  }
  output_texture.write(sum / kernel_size, xy);
}

[[ kernel ]]
void hbox_blur_x1(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                  texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                  constant int& radius [[ buffer(0) ]],
                  constant int& kernel_size [[ buffer(1) ]],
                  uint row [[ thread_position_in_grid ]]) {
  hbox_blur<1>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x1_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                    texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                    constant int& radius [[ buffer(0) ]],
                    constant int& kernel_size [[ buffer(1) ]],
                    uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<1>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x2(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                  texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                  constant int& radius [[ buffer(0) ]],
                  constant int& kernel_size [[ buffer(1) ]],
                  uint row [[ thread_position_in_grid ]]) {
  hbox_blur<2>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x2_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                    texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                    constant int& radius [[ buffer(0) ]],
                    constant int& kernel_size [[ buffer(1) ]],
                    uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<2>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x4(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                  texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                  constant int& radius [[ buffer(0) ]],
                  constant int& kernel_size [[ buffer(1) ]],
                  uint row [[ thread_position_in_grid ]]) {
  hbox_blur<4>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x4_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                    texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                    constant int& radius [[ buffer(0) ]],
                    constant int& kernel_size [[ buffer(1) ]],
                    uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<4>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x8(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                  texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                  constant int& radius [[ buffer(0) ]],
                  constant int& kernel_size [[ buffer(1) ]],
                  uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<8>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x8_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                    texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                    constant int& radius [[ buffer(0) ]],
                    constant int& kernel_size [[ buffer(1) ]],
                    uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<8>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x12(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                   texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                   constant int& radius [[ buffer(0) ]],
                   constant int& kernel_size [[ buffer(1) ]],
                   uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<12>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x12_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                        texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                        constant int& radius [[ buffer(0) ]],
                        constant int& kernel_size [[ buffer(1) ]],
                        uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<12>(input_texture, output_texture, radius, kernel_size, row);
}


[[ kernel ]]
void hbox_blur_x16(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                      texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                      constant int& radius [[ buffer(0) ]],
                      constant int& kernel_size [[ buffer(1) ]],
                      uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<16>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x16_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                      texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                      constant int& radius [[ buffer(0) ]],
                      constant int& kernel_size [[ buffer(1) ]],
                      uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<16>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x24(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                      texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                      constant int& radius [[ buffer(0) ]],
                      constant int& kernel_size [[ buffer(1) ]],
                      uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<24>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x24_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                      texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                      constant int& radius [[ buffer(0) ]],
                      constant int& kernel_size [[ buffer(1) ]],
                      uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<24>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x32(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                      texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                      constant int& radius [[ buffer(0) ]],
                      constant int& kernel_size [[ buffer(1) ]],
                      uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<32>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void hbox_blur_x32_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                      texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                      constant int& radius [[ buffer(0) ]],
                      constant int& kernel_size [[ buffer(1) ]],
                      uint row [[ thread_position_in_grid ]]) {
  
  hbox_blur<32>(input_texture, output_texture, radius, kernel_size, row);
}

[[ kernel ]]
void vbox_blur_x1(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                     texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<1>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x1_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                     texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<1>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x2(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                     texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<2>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x2_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                     texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<2>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x4(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                     texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<4>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x4_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                     texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<4>(input_texture, output_texture, radius, kernel_size, column);
}


[[ kernel ]]
void vbox_blur_x8(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                     texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<8>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x8_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                     texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<8>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x12(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                     texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<12>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x12_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                     texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<12>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x16(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                     texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<16>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x16_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                     texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<16>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x24(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                     texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<24>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x24_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                     texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<24>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x32(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                     texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<32>(input_texture, output_texture, radius, kernel_size, column);
}

[[ kernel ]]
void vbox_blur_x32_h(texture2d<half, access::sample> input_texture [[ texture(0) ]],
                     texture2d<half, access::read_write> output_texture [[ texture(1) ]],
                     constant int& radius [[ buffer(0) ]],
                     constant int& kernel_size [[ buffer(1) ]],
                  uint column [[ thread_position_in_grid ]]) {
  vbox_blur<32>(input_texture, output_texture, radius, kernel_size, column);
}
