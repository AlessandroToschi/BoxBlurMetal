//
//  main.swift
//  BoxBlurGPUTrace
//
//  Created by Alessandro Toschi on 27/02/24.
//

import Foundation
import BoxBlur
import Metal
import MetalKit

extension BoxBlur.PixelFormat: CustomStringConvertible {
  public var description: String {
    switch self {
      case .rgba8: "RGBA8"
      case .rgba16: "RGBA16"
      case .rgba32: "RGBA32"
    }
  }
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

let radius = 30
let runs = 10

do {
  let gpuTraceUrl = Bundle.main.executableURL!.deletingLastPathComponent().appending(path: "MPSTrace.gputrace")
  
  if FileManager.default.fileExists(atPath: gpuTraceUrl.path(percentEncoded: false)) {
    try! FileManager.default.removeItem(at: gpuTraceUrl)
  }
  
  let captureDescriptor = MTLCaptureDescriptor()
  captureDescriptor.captureObject = commandQueue
  captureDescriptor.destination = .gpuTraceDocument
  captureDescriptor.outputURL = gpuTraceUrl
  
  let captureManager = MTLCaptureManager.shared()
  try! captureManager.startCapture(with: captureDescriptor)
  
  let boxBlur = BoxBlur(
    radius: radius,
    device: device,
    intermediaTexturePixelFormat: .rgba32
  )
  boxBlur.load()
  
  for run in 1 ... runs {
    let commandBuffer = commandQueue.makeCommandBuffer()!
    commandBuffer.label = "Run \(run)"
    commandBuffer.enqueue()
    
    boxBlur.mps(
      commandBuffer: commandBuffer,
      inputTexture: inputTexture,
      outputTexture: outputTexture
    )
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }
  
  captureManager.stopCapture()
  
  print(gpuTraceUrl.absoluteString)
}

for pixelFormat in BoxBlur.PixelFormat.allCases {
  let gpuTraceUrl = Bundle.main.executableURL!.deletingLastPathComponent().appending(path: "GPUTrace\(pixelFormat).gputrace")
  
  if FileManager.default.fileExists(atPath: gpuTraceUrl.path(percentEncoded: false)) {
    try! FileManager.default.removeItem(at: gpuTraceUrl)
  }
  
  let captureDescriptor = MTLCaptureDescriptor()
  captureDescriptor.captureObject = commandQueue
  captureDescriptor.destination = .gpuTraceDocument
  captureDescriptor.outputURL = gpuTraceUrl
  
  let captureManager = MTLCaptureManager.shared()
  try! captureManager.startCapture(with: captureDescriptor)
  
  let boxBlur = BoxBlur(
    radius: radius,
    device: device,
    intermediaTexturePixelFormat: pixelFormat
  )
  boxBlur.load()
  
  for run in 1 ... runs {
    let commandBuffer = commandQueue.makeCommandBuffer()!
    commandBuffer.label = "Run \(run)"
    commandBuffer.enqueue()
    
    boxBlur.fastDoublePass(
      commandBuffer: commandBuffer,
      inputTexture: inputTexture,
      outputTexture: outputTexture,
      itemsPerKernel: .eight
    )
    
    boxBlur.mps(
      commandBuffer: commandBuffer,
      inputTexture: inputTexture,
      outputTexture: outputTexture
    )
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }
  
  captureManager.stopCapture()
  
  print(gpuTraceUrl.absoluteString)
}
