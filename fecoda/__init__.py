import logging
from mpi4py import MPI

logger = logging.getLogger("fecoda")
logger.setLevel(logging.INFO)

if MPI.COMM_WORLD.rank == 0:
    fh = logging.FileHandler("fecoda.log", mode="w")
    logger.addHandler(fh)
