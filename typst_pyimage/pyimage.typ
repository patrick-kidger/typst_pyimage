#let (pyinit, pyimage, pycontent) = {
  if sys.inputs.at("typst_pyimage.query", default: "false") == "true" {

    let pyinit(program) = {
      metadata("typst_pyimage.pyinit." + program)
    }
    let pyimage(program, ..arguments) = {
      metadata("typst_pyimage.pyimage." + program)
    }
    let pycontent(program, ..arguments) = {
      metadata("typst_pyimage.pycontent." + program)
    }
    (pyinit, pyimage, pycontent)

  } else {

    let mapping =  json("mapping.json")
    let call(program, fn) = {
      let num_matches = 0
      for (program_text, output_file) in mapping {
        if program.match(program_text) != none {
          num_matches = num_matches + 1
          fn(output_file)
        }
      }
      if num_matches != 1 {
        panic("Got " + str(num_matches) + " pyimage matches.")
      }
    }

    let pyinit(program) = {}
    let pyimage(program, ..arguments) = {
      call(program, image.with(..arguments))
    }
    let pycontent(program, ..arguments) = {
      call(program, output_file => eval(read(output_file, ..arguments), mode: "markup"))
    }
    (pyinit, pyimage, pycontent)

  }
}
