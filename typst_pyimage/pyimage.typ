#let pyimage_counter = counter("pyimage")
#let pycontent_counter = counter("pycontent")
#let pyinit(program) = {}
#let pyimage(program, ..arguments) = {
  pyimage_counter.step()
  locate(loc => image(str(pyimage_counter.at(loc).at(0)) + ".png", ..arguments))
}
#let pycontent(program) = {
  pycontent_counter.step()
  locate(loc => eval(read(str(pycontent_counter.at(loc).at(0)) + ".txt")))
}
