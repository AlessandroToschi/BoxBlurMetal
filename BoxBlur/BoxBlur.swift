//
//  BoxBlur.swift
//  BoxBlur
//
//  Created by Alessandro Toschi on 27/02/24.
//

import Foundation
import Metal
import MetalPerformanceShaders

public class BoxBlur {
  public enum ItemsPerKernel: Int, CaseIterable {
    case one = 1
    case two = 2
    case four = 4
    case eight = 8
    case twelve = 12
    case sixteen = 16
    case twentyfour = 24
    case thirtytwo = 32
  }
  public enum PixelFormat: CaseIterable {
    case rgba8
    case rgba16
    case rgba32
  }
  
  public var radius: Int
  public var kernelSize: Int {
    2 * self.radius + 1
  }
  
  private let device: MTLDevice
  private let intermediateTexturePixelFormat: PixelFormat
  
  private var hboxPipelineStates: [ItemsPerKernel: MTLComputePipelineState]!
  private var vboxPipelineStates: [ItemsPerKernel: MTLComputePipelineState]!
  private var singlePassPipelineState: MTLComputePipelineState!
  private var doublePassHPipelineState: MTLComputePipelineState!
  private var doublePassVPipelineState: MTLComputePipelineState!
  private var mpsImageBox: MPSImageBox!
  private var intermediateTexture: MTLTexture?
  
  public init(
    radius: Int,
    device: MTLDevice,
    intermediaTexturePixelFormat: PixelFormat
  ) {
    self.radius = radius
    self.device = device
    self.intermediateTexturePixelFormat = intermediaTexturePixelFormat
  }
  
  public func load() {
    let library = try! self.device.makeDefaultLibrary(bundle: Bundle(for: Self.self))
    
    let computePipelineDescriptor = MTLComputePipelineDescriptor()
    computePipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
    
    self.hboxPipelineStates = [:]
    self.vboxPipelineStates = [:]
    
    for itemsPerKernel in ItemsPerKernel.allCases {
      let hFunctionNames = ["hbox_blur_x\(itemsPerKernel.rawValue)", "hbox_blur_x\(itemsPerKernel.rawValue)_h"]
      let vFunctionNames = ["vbox_blur_x\(itemsPerKernel.rawValue)", "vbox_blur_x\(itemsPerKernel.rawValue)_h"]
      
      for hFunctionName in hFunctionNames {
        let function = library.makeFunction(name: hFunctionName)!
        
        computePipelineDescriptor.computeFunction = function
        computePipelineDescriptor.label = function.name
        
        self.hboxPipelineStates[itemsPerKernel] = try! self.device.makeComputePipelineState(
          descriptor: computePipelineDescriptor,
          options: []
        ).0
      }
      
      for vFunctionName in vFunctionNames {
        let function = library.makeFunction(name: vFunctionName)!
        
        computePipelineDescriptor.computeFunction = function
        computePipelineDescriptor.label = function.name
        
        self.vboxPipelineStates[itemsPerKernel] = try! self.device.makeComputePipelineState(
          descriptor: computePipelineDescriptor,
          options: []
        ).0
      }
    }
    
    let singlePassFunction = library.makeFunction(name: "box_blur_single_pass")!
    computePipelineDescriptor.computeFunction = singlePassFunction
    computePipelineDescriptor.label = singlePassFunction.name
    self.singlePassPipelineState = try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0
    
    let doublePassHFunction = library.makeFunction(name: "box_blur_double_pass_h")!
    let doublePassVFunction = library.makeFunction(name: "box_blur_double_pass_v")!
    
    computePipelineDescriptor.computeFunction = doublePassHFunction
    computePipelineDescriptor.label = doublePassHFunction.name
    self.doublePassHPipelineState = try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0
    
    computePipelineDescriptor.computeFunction = doublePassVFunction
    computePipelineDescriptor.label = doublePassVFunction.name
    self.doublePassVPipelineState = try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: []).0
    
    self.mpsImageBox = MPSImageBox(
      device: self.device,
      kernelWidth: self.kernelSize,
      kernelHeight: self.kernelSize
    )
    self.mpsImageBox.edgeMode = .clamp
  }
  
  private func createIntermediateTextureIfNeeded(width: Int, height: Int) {
    if let intermediateTexture, intermediateTexture.width == width, intermediateTexture.height == height {
      return
    }
    
    let pixelFormat: MTLPixelFormat = switch self.intermediateTexturePixelFormat {
    case .rgba8: .rgba8Unorm
    case .rgba16: .rgba16Float
    case .rgba32: .rgba32Float
    }
    
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: pixelFormat,
      width: width,
      height: height,
      mipmapped: false
    )
    textureDescriptor.storageMode = .private
    textureDescriptor.usage = [.shaderRead, .shaderWrite]
    
    self.intermediateTexture = self.device.makeTexture(descriptor: textureDescriptor)!
  }
  
  public func singlePass(
    commandBuffer: MTLCommandBuffer,
    inputTexture: MTLTexture,
    outputTexture: MTLTexture
  ) {
    var computeRadius: Int32 = Int32(self.radius)
    var computeKernelSize: Int32 = Int32(self.kernelSize)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.label = self.singlePassPipelineState.label
    computeCommandEncoder.setComputePipelineState(self.singlePassPipelineState)
    computeCommandEncoder.setTexture(inputTexture, index: 0)
    computeCommandEncoder.setTexture(outputTexture, index: 1)
    computeCommandEncoder.setBytes(&computeRadius, length: MemoryLayout.stride(ofValue: computeRadius), index: 0)
    computeCommandEncoder.setBytes(&computeKernelSize, length: MemoryLayout.stride(ofValue: computeKernelSize), index: 1)
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
  }
  
  public func doublePass(
    commandBuffer: MTLCommandBuffer,
    inputTexture: MTLTexture,
    outputTexture: MTLTexture
  ) {
    var computeRadius: Int32 = Int32(self.radius)
    var computeKernelSize: Int32 = Int32(self.kernelSize)
    
    self.createIntermediateTextureIfNeeded(
      width: inputTexture.width,
      height: inputTexture.height
    )
    
    let threasPerGrid = MTLSize(
      width: inputTexture.width,
      height: inputTexture.height,
      depth: 1
    )
    
    let threadsPerThreadgroup = MTLSize(
      width: doublePassHPipelineState.threadExecutionWidth,
      height: doublePassHPipelineState.maxTotalThreadsPerThreadgroup / singlePassPipelineState.threadExecutionWidth,
      depth: 1
    )
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.label = self.doublePassHPipelineState.label
    computeCommandEncoder.setComputePipelineState(self.doublePassHPipelineState)
    computeCommandEncoder.setTexture(inputTexture, index: 0)
    computeCommandEncoder.setTexture(self.intermediateTexture, index: 1)
    computeCommandEncoder.setBytes(&computeRadius, length: MemoryLayout.stride(ofValue: computeRadius), index: 0)
    computeCommandEncoder.setBytes(&computeKernelSize, length: MemoryLayout.stride(ofValue: computeKernelSize), index: 1)
    computeCommandEncoder.dispatchThreads(
      threasPerGrid,
      threadsPerThreadgroup: threadsPerThreadgroup
    )
    computeCommandEncoder.label = self.doublePassVPipelineState.label
    computeCommandEncoder.setComputePipelineState(self.doublePassVPipelineState)
    computeCommandEncoder.setTexture(self.intermediateTexture, index: 0)
    computeCommandEncoder.setTexture(outputTexture, index: 1)
    computeCommandEncoder.dispatchThreads(
      threasPerGrid,
      threadsPerThreadgroup: threadsPerThreadgroup
    )
    computeCommandEncoder.endEncoding()
  }
  
  public func fastDoublePass(
    commandBuffer: MTLCommandBuffer,
    inputTexture: MTLTexture,
    outputTexture: MTLTexture,
    itemsPerKernel: ItemsPerKernel
  ) {
    var computeRadius: Int32 = Int32(self.radius)
    var computeKernelSize: Int32 = Int32(self.kernelSize)
    
    self.createIntermediateTextureIfNeeded(
      width: inputTexture.width,
      height: inputTexture.height
    )
    
    let hboxPipelineState = self.hboxPipelineStates[itemsPerKernel]!
    let vboxPipelineState = self.vboxPipelineStates[itemsPerKernel]!
    
    let threadsPerThreadgroup = MTLSize(
      width: hboxPipelineState.threadExecutionWidth,
      height: 1,
      depth: 1
    )
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.label = hboxPipelineState.label
    computeCommandEncoder.setComputePipelineState(hboxPipelineState)
    computeCommandEncoder.setTexture(inputTexture, index: 0)
    computeCommandEncoder.setTexture(self.intermediateTexture, index: 1)
    computeCommandEncoder.setBytes(&computeRadius, length: MemoryLayout.stride(ofValue: computeRadius), index: 0)
    computeCommandEncoder.setBytes(&computeKernelSize, length: MemoryLayout.stride(ofValue: computeKernelSize), index: 1)
    computeCommandEncoder.dispatchThreads(
      MTLSize(width: inputTexture.height, height: 1, depth: 1),
      threadsPerThreadgroup: threadsPerThreadgroup
    )
    computeCommandEncoder.label = vboxPipelineState.label
    computeCommandEncoder.setComputePipelineState(vboxPipelineState)
    computeCommandEncoder.setTexture(self.intermediateTexture, index: 0)
    computeCommandEncoder.setTexture(outputTexture, index: 1)
    computeCommandEncoder.dispatchThreads(
      MTLSize(width: inputTexture.width, height: 1, depth: 1),
      threadsPerThreadgroup: threadsPerThreadgroup
    )
    computeCommandEncoder.endEncoding()
  }
  
  public func mps(
    commandBuffer: MTLCommandBuffer,
    inputTexture: MTLTexture,
    outputTexture: MTLTexture
  ) {
    self.mpsImageBox.encode(
      commandBuffer: commandBuffer,
      sourceTexture: inputTexture,
      destinationTexture: outputTexture
    )
  }
}
