#let pyinit(program) = {
  metadata("typst_pyimage.pyinit." + program)
}
#let mapping = json("mapping.json")
#let pyimage(program, ..arguments) = {
  metadata("typst_pyimage.pyimage." + program)
  let i = 0
  for (program_text, output_file) in mapping {
    if program.match(program_text) != none {
      i = i + 1
      image(output_file, ..arguments)
    }
  }
  if i != 1 {
    panic("Got " + str(i) + " pyimage matches.")
  }
}
#let pycontent(program, ..arguments) = {
  metadata("typst_pyimage.pycontent." + program)
  let i = 0
  for (program_text, output_file) in mapping {
    if program.match(program_text) != none {
      i = i + 1
      eval(read(output_file, ..arguments), mode: "markup")
    }
  }
  if i != 1 {
    panic("Got " + str(i) + " pyimage matches.")
  }
}
