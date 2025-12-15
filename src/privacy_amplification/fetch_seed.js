import { Client, DIRNGClient } from '@buff-beacon-project/curby-client'
import { appendFile } from 'fs/promises'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __filename = fileURLToPath(import.meta.url) // absolute path of the file currently being executed
const __dirname = dirname(__filename) // directory containing that file

// Reference: https://random.colorado.edu/

// Fetch a classical beacon pulse and turn it into bits (a binary string)
async function randomness_from_classical_beacon(client) {
  const randomness = await client.randomness()
  //console.log(randomness)

  const bytes = randomness.bytes()
  //console.log("bytes:", bytes)

  // Convert each byte to an 8-bit binary string
  const bitString = Array.from(bytes)
      .map(b => b.toString(2).padStart(8, "0"))
      .join("");

  console.log("bitString:", bitString)
  //console.log("length:", bitString.length)

  const outputPath = join(__dirname, "seeds.txt")
  // Save bitstring (512 bits) to output file: one line per pulse is appended to the file
  await appendFile(outputPath, bitString + "\n")
  console.log("Saved 512-bit string to:", outputPath)
}

// Fetch a quantum beacon pulse and turn it into bits (a binary string)
// The CURBy Quantum Randomness generation is currently offline for relocation and upgrades.
// Quantum randomness will be unavailable until the upgrade is complete, which is expected to be in early 2026. 
async function randomness_from_quantum_beacon() {
  const dirng = DIRNGClient.create()
  const quantumRandomness = await dirng.randomness()
  //console.log(quantumRandomness)

  const bytes = quantumRandomness.bytes()
  //console.log("bytes:", bytes)

  // Convert each byte to an 8-bit binary string
  const bitString = Array.from(bytes)
      .map(b => b.toString(2).padStart(8, "0"))
      .join("");

  console.log("bitString:", bitString)
  //console.log("length:", bitString.length)
}

async function main() {
    const client = Client.create()
    for (let i = 0; i < 30; i++) {
        await client.waitForNext() // Wait for the next pulse
        console.log(`\n=== Fetching pulse ${i + 1} ===`)
        await randomness_from_classical_beacon(client)
  }
}

main()