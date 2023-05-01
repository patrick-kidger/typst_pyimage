#let pyimage_counter = counter("pyimage")
#let pyimage(program, ..arguments) = {
  pyimage_counter.step()
  locate(loc => image(str(pyimage_counter.at(loc).at(0)) + ".png", ..arguments))
}
#let pyimageinit(program) = {}
