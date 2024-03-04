//
//  main.swift
//  BoxBlurRunStats
//
//  Created by Alessandro Toschi on 27/02/24.
//

import Foundation
import BoxBlur
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

struct RunStat: Hashable, Equatable {
  let label: String
  let radius: Int
  let elapsedTime: CFTimeInterval
  let pixelFormat: BoxBlur.PixelFormat
  
  var csvString: String {
    "\(label);\(radius);\(elapsedTime);\(elapsedTime * 1000.0);\(pixelFormat);\n"
  }
  
  static var csvHeader: String {
    "label;radius;time(s);time(ms);pixelFormat;\n"
  }
}

func writeRunStats() throws {
  let statsUrl = Bundle.main.executableURL!.deletingLastPathComponent().appending(path: "stats.csv")
  try RunStat.csvHeader.write(to: statsUrl, atomically: true, encoding: .utf8)
  
  let fileHandle = try FileHandle(forWritingTo: statsUrl)
  try fileHandle.seekToEnd()
  
  for runStat in runStats {
    try fileHandle.write(contentsOf: runStat.csvString.data(using: .utf8)!)
  }
  
  try fileHandle.close()
  
  print(statsUrl.absoluteString)
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

let radii = 1 ... 63
let runsPerRadius = 10

func encodeInCommandBuffer(label: String, callback: (MTLCommandBuffer) -> ()) -> CFTimeInterval {
  let commandBuffer = commandQueue.makeCommandBuffer()!
  commandBuffer.label = label
  commandBuffer.enqueue()
  callback(commandBuffer)
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
}

var runStats = [RunStat]()

for pixelFormat in BoxBlur.PixelFormat.allCases {
  for radius in radii {
    print("Radius: \(radius) - PixelFormat: \(pixelFormat)")
    let boxBlur = BoxBlur(
      radius: radius,
      device: device,
      intermediaTexturePixelFormat: pixelFormat
    )
    boxBlur.load()
    for run in 1 ... runsPerRadius {
      print("Run \(run)...")
      runStats.append(
        RunStat(
          label: "single_pass",
          radius: radius,
          elapsedTime: encodeInCommandBuffer(
            label: "single_pass",
            callback: {
              commandBuffer in
              boxBlur.singlePass(
                commandBuffer: commandBuffer,
                inputTexture: inputTexture,
                outputTexture: outputTexture
              )
            }
          ),
          pixelFormat: pixelFormat
        )
      )
      
      runStats.append(
        RunStat(
          label: "double_pass",
          radius: radius,
          elapsedTime: encodeInCommandBuffer(
            label: "double_pass",
            callback: {
              commandBuffer in
              boxBlur.doublePass(
                commandBuffer: commandBuffer,
                inputTexture: inputTexture,
                outputTexture: outputTexture
              )
            }
          ),
          pixelFormat: pixelFormat
        )
      )
      
      for itemsPerKernel in BoxBlur.ItemsPerKernel.allCases {
        let label = "fast_doube_pass_x\(itemsPerKernel)"
        runStats.append(
          RunStat(
            label: label,
            radius: radius,
            elapsedTime: encodeInCommandBuffer(
              label: label,
              callback: {
                commandBuffer in
                boxBlur.fastDoublePass(
                  commandBuffer: commandBuffer,
                  inputTexture: inputTexture,
                  outputTexture: outputTexture,
                  itemsPerKernel: itemsPerKernel
                )
              }
            ),
            pixelFormat: pixelFormat
          )
        )
      }
      
      runStats.append(
        RunStat(
          label: "MPS",
          radius: radius,
          elapsedTime: encodeInCommandBuffer(
            label: "MPS",
            callback: {
              commandBuffer in
              boxBlur.mps(
                commandBuffer: commandBuffer,
                inputTexture: inputTexture,
                outputTexture: outputTexture
              )
            }
          ),
          pixelFormat: pixelFormat
        )
      )
    }
  }
}

try! writeRunStats()
