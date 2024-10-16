from netgen.occ import MoveTo, Box, Glue, OCCGeometry, X, Y, Z
# Not sure why but it seems that the import of netgen.gui is necessary for the GUI to start
import netgen.gui # pylint: disable=unused-import
import ngsolve
from rf_solver import SolverRF

def create_geometry():
    cm = 1e-2
    mm = 1e-3

    sw = 70 * mm
    sl = 70 * mm
    sh = 1.6 * mm
    pw = 49.4 * mm
    pl = 41.3 * mm
    wf = 4.95 * mm
    il = 15.7 * mm
    iw = 0.8 * mm

    air_l = 10 * cm
    air_w = 10 * cm
    air_h = 4 * cm

    ground_copper = MoveTo(-sw / 2, -sl / 2).Rectangle(sw, sl).Face()
    ground_copper.faces.col = (1, 0, 0)

    dielectric = MoveTo(-sw / 2, -sl / 2).Rectangle(sw, sl).Face().Extrude(sh)

    top_copper = (
        MoveTo(-pw / 2, -pl / 2)
        .Line(pw / 2 - iw - wf / 2)
        .Rotate(90)
        .Line(il)
        .Rotate(-90)
        .Line(iw)
        .Rotate(-90)
        .Line(il + (sl - pl) / 2)
        .Rotate(90)
        .Line(wf)
        .Rotate(90)
        .Line(il + (sl - pl) / 2)
        .Rotate(-90)
        .Line(iw)
        .Rotate(-90)
        .Line(il)
        .Rotate(90)
        .Line(pw / 2 - iw - wf / 2)
        .Rotate(90)
        .Line(pl)
        .Rotate(90)
        .Line(pw)
        .Close()
        .Face()
        .Move((0, 0, sh))
    )
    top_copper.faces.col = (0, 0, 1)

    air_box = Box((-air_w / 2, -air_l / 2, -air_h / 2), (air_w / 2, air_l / 2, air_h / 2))
    air_box.faces.name = "air_outer"
    air_box = air_box - dielectric
    air_box.faces.col = (0, 0, 1, 0.3)
    air_box.solids.name = "air"
    ground_copper.faces.name = "ground"
    top_copper.faces.name = "patch"
    dielectric.solids.name = "dielectric"
    dielectric.faces.col = (1, 1, 0)


    pins = Box((0, -sl / 2, 0), (wf / 2, -sl / 2 + il / 2, sh))
    pins.edges.Min(X + Y).name = "loading_pin"

    # mesh size presets
    #ground_copper.faces.maxh = 2 * sh
    #dielectric.faces.maxh = 2 * sh
    #top_copper.faces.maxh = 2 * sh
    #pins.edges["loading_pin"].maxh = sh / 2

    faces_pins = pins.faces[Z > 0 - 1e-6][Z < sh + 1e-6]
    shape = Glue([dielectric, ground_copper, top_copper] + list(faces_pins))

    shape = Glue([shape, air_box])

    geom = OCCGeometry(shape)

    return geom

if __name__ == "__main__":
    geom = create_geometry()
    ngmesh = geom.GenerateMesh(maxh=0.003, segmentsperedge=0)
    mesh = ngsolve.Mesh(ngmesh)
    #ngsolve.Draw(mesh)


    properties = {}
    properties["frequency"] = 2.4e9

    domains = {"dielectric": {"type": "dielectric", "epsilon": 2.2}}
    boundaries = {"air_outer": {"type": "outer"}, "ground": {"type": "pec"}, "patch": {"type": "pec"}}
    lumped_elements = {"loading_pin": {"type": "feed"}}

    print(lumped_elements)
    solver = SolverRF(mesh, domains, boundaries, lumped_elements, properties)
    solver.assemble()
    solver.solve()
    solver.finalize()
    #ngsolve.Draw(mesh)
    #input()
