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

// MARK: Compute Pipelines
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

// MARK: Single Pass
let singlePassFunction = library.makeFunction(name: "box_blur_single_pass")!
computePipelineDescriptor.computeFunction = singlePassFunction
computePipelineDescriptor.label = singlePassFunction.name
let singlePassPipelineState = try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0

// MARK: Parameters
var radius: Int32 = 30
var kernelSize: Int32 = 2 * radius + 1
let run = 5

// MARK: MPSImageBox
let mpsBoxFilter = MPSImageBox(
  device: device,
  kernelWidth: Int(kernelSize),
  kernelHeight: Int(kernelSize)
)
mpsBoxFilter.edgeMode = .clamp


// MARK: Metal Capture
#if DEBUG
let gpuTraceUrl = Bundle.main.executableURL!.deletingLastPathComponent().appending(path: "boxblur.gputrace")
try? FileManager.default.removeItem(at: gpuTraceUrl)

let captureDescriptor = MTLCaptureDescriptor()
captureDescriptor.captureObject = commandQueue
captureDescriptor.destination = .gpuTraceDocument
captureDescriptor.outputURL = gpuTraceUrl

let captureDevice = MTLCaptureManager.shared()
try! captureDevice.startCapture(with: captureDescriptor)
#endif

// MARK: Run

for i in 0 ..< run {
  let commandBuffer = commandQueue.makeCommandBuffer()!
  commandBuffer.label = "Box Blur \(i + 1)"
  commandBuffer.enqueue()
  
  let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
  computeCommandEncoder.setBytes(&radius, length: MemoryLayout<Int32>.stride, index: 0)
  computeCommandEncoder.setBytes(&kernelSize, length: MemoryLayout<Int32>.stride, index: 1)
  
  for (hboxPipelineState, vboxPipelineState) in zip(hboxPipelineStates, vboxPipelineStates) {
    computeCommandEncoder.label = "HBox \(i + 1)"
    computeCommandEncoder.setTexture(inputTexture, index: 0)
    computeCommandEncoder.setTexture(tempTexture, index: 1)
    computeCommandEncoder.setComputePipelineState(hboxPipelineState)
    computeCommandEncoder.dispatchThreads(
      MTLSizeMake(inputTexture.height, 1, 1),
      threadsPerThreadgroup: MTLSize(width: hboxPipelineState.threadExecutionWidth, height: 1, depth: 1)
    )
    computeCommandEncoder.label = "VBox \(i + 1)"
    computeCommandEncoder.setTexture(tempTexture, index: 0)
    computeCommandEncoder.setTexture(outputTexture, index: 1)
    computeCommandEncoder.setComputePipelineState(vboxPipelineState)
    computeCommandEncoder.dispatchThreads(
      MTLSizeMake(inputTexture.width, 1, 1),
      threadsPerThreadgroup: MTLSize(width: vboxPipelineState.threadExecutionWidth, height: 1, depth: 1)
    )
  }
  
  computeCommandEncoder.label = "Single Pass \(i + 1)"
  computeCommandEncoder.setComputePipelineState(singlePassPipelineState)
  computeCommandEncoder.setTexture(inputTexture, index: 0)
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

  mpsBoxFilter.encode(
    commandBuffer: commandBuffer,
    sourceTexture: inputTexture,
    destinationTexture: outputTexture
  )
  
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
}

#if DEBUG
captureDevice.stopCapture()

print(captureDescriptor.outputURL!)
#endif
