import strawberryfields as sf
from strawberryfields.ops import *

prog = sf.Program(2)

with prog.context as q:
    Sgate(1.0) | q[0]

eng = sf.LocalEngine(backend ='gaussian')
results = eng.run(prog)