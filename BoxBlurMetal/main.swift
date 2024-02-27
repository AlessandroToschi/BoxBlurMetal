//
//  main.swift
//  BoxBlurMetal
//
//  Created by Alessandro Toschi on 25/02/24.
//

import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders

struct RunStat: Hashable, Equatable {
  let label: String
  let radius: Int
  let elapsedTime: CFTimeInterval
}

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

// MARK: Input Texture
let inputTextureUrl = Bundle.main.url(forResource: "IMG_5102", withExtension: "jpg")!
let textureLoader = MTKTextureLoader(device: device)
let inputTexture = try! textureLoader.newTexture(
  URL: inputTextureUrl,
  options: [
    .textureStorageMode: MTLStorageMode.shared.rawValue,
    .textureUsage: MTLTextureUsage.shaderRead.rawValue
  ]
)

// MARK: Output Texture
let outputTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
  pixelFormat: .bgra8Unorm,
  width: inputTexture.width,
  height: inputTexture.height,
  mipmapped: false
)
outputTextureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
outputTextureDescriptor.storageMode = .private
let outputTexture = device.makeTexture(descriptor: outputTextureDescriptor)!

// MARK: Temporary Texture
let tempTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
  pixelFormat: .rgba8Unorm,
  width: inputTexture.width,
  height: inputTexture.height,
  mipmapped: false
)
tempTextureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
tempTextureDescriptor.storageMode = .private
let tempTexture = device.makeTexture(descriptor: tempTextureDescriptor)!

// MARK: HBox VBox Compute Pipelines
let computePipelineDescriptor = MTLComputePipelineDescriptor()
computePipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true

let library = device.makeDefaultLibrary()!
let itemsPerInvocation = [1, 2, 4, 8, 12, 16, 24, 32]
let hboxFunctions = itemsPerInvocation.map {
  let functionName = "hbox_blur_x\($0)"
  return [
    library.makeFunction(name: functionName)!,
    library.makeFunction(name: functionName + "_h")!
  ]
}.flatMap{ $0 }
let vboxFunctions = itemsPerInvocation.map {
  let functionName = "vbox_blur_x\($0)"
  return [
    library.makeFunction(name: functionName)!,
    library.makeFunction(name: functionName + "_h")!
  ]
}.flatMap{ $0 }
let hboxPipelineStates = hboxFunctions.map {
  computePipelineDescriptor.computeFunction = $0
  computePipelineDescriptor.label = $0.name
  return try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0
}
let vboxPipelineStates = vboxFunctions.map {
  computePipelineDescriptor.computeFunction = $0
  computePipelineDescriptor.label = $0.name
  return try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0
}

func hvBox(
  run: Int,
  radius: Int,
  kernelSize: Int,
  hboxPipelineState: MTLComputePipelineState,
  vboxPipelineState: MTLComputePipelineState
) -> RunStat {
  var computeRadius: Int32 = Int32(radius)
  var computeKernelSize: Int32 = Int32(kernelSize)
  
  let label = hboxPipelineState.label!.split(separator: "_")[2...].joined(separator: "_")
  let commandBuffer = commandQueue.makeCommandBuffer()!
  commandBuffer.label = "Double Pass \(run) - \(label)"
  commandBuffer.enqueue()
  
  let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
  computeCommandEncoder.setBytes(&computeRadius, length: MemoryLayout.stride(ofValue: computeRadius), index: 0)
  computeCommandEncoder.setBytes(&computeKernelSize, length: MemoryLayout.stride(ofValue: computeKernelSize), index: 1)
  computeCommandEncoder.label = "HBox \(run) \(label)"
  computeCommandEncoder.setTexture(inputTexture, index: 0)
  computeCommandEncoder.setTexture(tempTexture, index: 1)
  computeCommandEncoder.setComputePipelineState(hboxPipelineState)
  computeCommandEncoder.dispatchThreads(
    MTLSizeMake(inputTexture.height, 1, 1),
    threadsPerThreadgroup: MTLSize(width: hboxPipelineState.threadExecutionWidth, height: 1, depth: 1)
  )
  computeCommandEncoder.label = "VBox \(run) \(label)"
  computeCommandEncoder.setTexture(tempTexture, index: 0)
  computeCommandEncoder.setTexture(outputTexture, index: 1)
  computeCommandEncoder.setComputePipelineState(vboxPipelineState)
  computeCommandEncoder.dispatchThreads(
    MTLSizeMake(inputTexture.width, 1, 1),
    threadsPerThreadgroup: MTLSize(width: vboxPipelineState.threadExecutionWidth, height: 1, depth: 1)
  )
  computeCommandEncoder.endEncoding()
  
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  
  let startTime = commandBuffer.gpuStartTime
  let endTime = commandBuffer.gpuEndTime
  let elapsed = endTime - startTime
  
  return RunStat(
    label: label,
    radius: radius,
    elapsedTime: elapsed
  )
}

// MARK: Single Pass
let singlePassFunction = library.makeFunction(name: "box_blur_single_pass")!
computePipelineDescriptor.computeFunction = singlePassFunction
computePipelineDescriptor.label = singlePassFunction.name
let singlePassPipelineState = try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0

// MARK: Double Pass
let doublePassHFunction = library.makeFunction(name: "box_blur_double_pass_h")!
let doublePassVFunction = library.makeFunction(name: "box_blur_double_pass_v")!
computePipelineDescriptor.computeFunction = doublePassHFunction
computePipelineDescriptor.label = doublePassHFunction.name
let doublePassHPipelineState = try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0
computePipelineDescriptor.computeFunction = doublePassVFunction
computePipelineDescriptor.label = doublePassVFunction.name
let doublePassVPipelineState = try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0

// MARK: Parameters
let radii = 1 ... 63
let runPerRadius = 10

// MARK: Run

var results = [String: [Int: [CFTimeInterval]]]()

for radius in radii {
  var computeRadius: Int32 = Int32(radius)
  var kernelSize  = 2 * computeRadius + 1
  print("Radius \(radius) - \(kernelSize)x\(kernelSize)")
  
  let mpsBoxFilter = MPSImageBox(
    device: device,
    kernelWidth: Int(kernelSize),
    kernelHeight: Int(kernelSize)
  )
  mpsBoxFilter.edgeMode = .clamp
  
  for run in 1 ... runPerRadius {
    print("Run \(run)...")
    for (hboxPipelineState, vboxPipelineState) in zip(hboxPipelineStates, vboxPipelineStates) {
      let label = hboxPipelineState.label!.split(separator: "_")[2...].joined(separator: "_")
      let commandBuffer = commandQueue.makeCommandBuffer()!
      commandBuffer.label = "Double Pass \(run) - \(label)"
      commandBuffer.enqueue()
      
      let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
      computeCommandEncoder.setBytes(&computeRadius, length: MemoryLayout<Int32>.stride, index: 0)
      computeCommandEncoder.setBytes(&kernelSize, length: MemoryLayout<Int32>.stride, index: 1)
      computeCommandEncoder.label = "HBox \(run) \(label)"
      computeCommandEncoder.setTexture(inputTexture, index: 0)
      computeCommandEncoder.setTexture(tempTexture, index: 1)
      computeCommandEncoder.setComputePipelineState(hboxPipelineState)
      computeCommandEncoder.dispatchThreads(
        MTLSizeMake(inputTexture.height, 1, 1),
        threadsPerThreadgroup: MTLSize(width: hboxPipelineState.threadExecutionWidth, height: 1, depth: 1)
      )
      computeCommandEncoder.label = "VBox \(run) \(label)"
      computeCommandEncoder.setTexture(tempTexture, index: 0)
      computeCommandEncoder.setTexture(outputTexture, index: 1)
      computeCommandEncoder.setComputePipelineState(vboxPipelineState)
      computeCommandEncoder.dispatchThreads(
        MTLSizeMake(inputTexture.width, 1, 1),
        threadsPerThreadgroup: MTLSize(width: vboxPipelineState.threadExecutionWidth, height: 1, depth: 1)
      )
      computeCommandEncoder.endEncoding()
      
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      
      let startTime = commandBuffer.gpuStartTime
      let endTime = commandBuffer.gpuEndTime
      let elapsed = endTime - startTime
      results[label, default: [:]][radius, default: []].append(elapsed)
    }
    
    var commandBuffer = commandQueue.makeCommandBuffer()!
    commandBuffer.label = "Single Pass \(run)"
    commandBuffer.enqueue()
    
    var computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBytes(&computeRadius, length: MemoryLayout<Int32>.stride, index: 0)
    computeCommandEncoder.setBytes(&kernelSize, length: MemoryLayout<Int32>.stride, index: 1)
    computeCommandEncoder.label = "Single Pass \(run)"
    computeCommandEncoder.setTexture(inputTexture, index: 0)
    computeCommandEncoder.setTexture(outputTexture, index: 1)
    computeCommandEncoder.setComputePipelineState(singlePassPipelineState)
    computeCommandEncoder.dispatchThreads(
      MTLSize(
        width: inputTexture.width,
        height: inputTexture.height,
        depth: 1
      ),
      threadsPerThreadgroup: MTLSize(
        width: singlePassPipelineState.threadExecutionWidth,
        height: singlePassPipelineState.maxTotalThreadsPerThreadgroup / singlePassPipelineState.threadExecutionWidth,
        depth: 1
      )
    )
    computeCommandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    var startTime = commandBuffer.gpuStartTime
    var endTime = commandBuffer.gpuEndTime
    var elapsed = endTime - startTime
    results["single_pass", default: [:]][radius, default: []].append(elapsed)
    
    commandBuffer = commandQueue.makeCommandBuffer()!
    commandBuffer.label = "Double Pass \(run)"
    commandBuffer.enqueue()
    
    computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBytes(&computeRadius, length: MemoryLayout<Int32>.stride, index: 0)
    computeCommandEncoder.setBytes(&kernelSize, length: MemoryLayout<Int32>.stride, index: 1)
    computeCommandEncoder.label = "Double Pass \(run)"
    computeCommandEncoder.setTexture(inputTexture, index: 0)
    computeCommandEncoder.setTexture(tempTexture, index: 1)
    computeCommandEncoder.setComputePipelineState(doublePassHPipelineState)
    computeCommandEncoder.dispatchThreads(
      MTLSize(
        width: inputTexture.width,
        height: inputTexture.height,
        depth: 1
      ),
      threadsPerThreadgroup: MTLSize(
        width: singlePassPipelineState.threadExecutionWidth,
        height: singlePassPipelineState.maxTotalThreadsPerThreadgroup / singlePassPipelineState.threadExecutionWidth,
        depth: 1
      )
    )
    computeCommandEncoder.setTexture(tempTexture, index: 0)
    computeCommandEncoder.setTexture(outputTexture, index: 1)
    computeCommandEncoder.dispatchThreads(
      MTLSize(
        width: inputTexture.width,
        height: inputTexture.height,
        depth: 1
      ),
      threadsPerThreadgroup: MTLSize(
        width: singlePassPipelineState.threadExecutionWidth,
        height: singlePassPipelineState.maxTotalThreadsPerThreadgroup / singlePassPipelineState.threadExecutionWidth,
        depth: 1
      )
    )
    computeCommandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    startTime = commandBuffer.gpuStartTime
    endTime = commandBuffer.gpuEndTime
    elapsed = endTime - startTime
    results["double_pass", default: [:]][radius, default: []].append(elapsed)
    
    commandBuffer = commandQueue.makeCommandBuffer()!
    commandBuffer.label = "MPSImageBox"
    commandBuffer.enqueue()

    mpsBoxFilter.encode(
      commandBuffer: commandBuffer,
      sourceTexture: inputTexture,
      destinationTexture: outputTexture
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    startTime = commandBuffer.gpuStartTime
    endTime = commandBuffer.gpuEndTime
    elapsed = endTime - startTime
    results["MPSImageBox", default: [:]][radius, default: []].append(elapsed)
  }
}
