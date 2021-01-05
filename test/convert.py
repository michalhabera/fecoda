import meshio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--infile", help="Gmsh .msh input file (with extension)")
parser.add_argument("--outfile",
                    help="Xdmf basename for output files (without extension)")
parser.add_argument("--prune_z", default=False, type=bool, help="Throw away z-component")
args = parser.parse_args()

mesh = meshio.read(args.infile)

if args.prune_z is True:
    print("Pruning z-component")
    mesh.points = mesh.points[:, 0:2]

cells = {}
cell_data = {}

entity_cells = {}
entity_data = {}

volume_cell = list(mesh.cells_dict.keys())[-1]
cells[volume_cell] = mesh.cells_dict[volume_cell]
print("Mesh of {} {}".format(len(cells[volume_cell]), volume_cell))

if volume_cell in mesh.cell_data_dict["gmsh:physical"]:
    cell_data[volume_cell] = [mesh.get_cell_data("gmsh:physical", volume_cell)]
    print("Tagged {} {}".format(len(cell_data[volume_cell][0]), volume_cell))
else:
    cell_data[volume_cell] = []

meshio.write("{}.xdmf".format(args.outfile),
             meshio.Mesh(points=mesh.points,
                         cells=cells,
                         cell_data=cell_data),
             file_format="xdmf")

if len(mesh.cells_dict.keys()) > 1:
    entity_cell = list(mesh.cells_dict.keys())[-2]
    entity_cells[entity_cell] = mesh.cells_dict[entity_cell]

    if entity_cell in mesh.cell_data_dict["gmsh:physical"]:
        entity_data[entity_cell] = [mesh.get_cell_data("gmsh:physical", entity_cell)]
        print("Tagged {} {}".format(len(entity_data[entity_cell][0]), entity_cell))
    else:
        entity_data[entity_cell] = []

    meshio.write("{}_{}.xdmf".format(args.outfile, entity_cell),
                 meshio.Mesh(points=mesh.points,
                             cells=entity_cells,
                             cell_data=entity_data),
                 file_format="xdmf")
