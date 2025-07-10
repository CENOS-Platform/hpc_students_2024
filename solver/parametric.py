from netgen.occ import MoveTo, Box, Glue, OCCGeometry, X, Y, Z
# It seems that the import of netgen.gui is necessary for the GUI to start
#import netgen.gui # pylint: disable=unused-import
import ngsolve
import matplotlib.pyplot as plt
import numpy as np
from rf_solver import SolverRF

class IfaAntennaCase:
    def __init__(
        self,
        feed_gap=0.8e-3,
        line_width=1e-3,
        substrate_eps=2.2,
        maxh=0.01
    ):
        self.feed_gap = feed_gap
        self.line_width = line_width
        self.substrate_eps = substrate_eps
        self.maxh = maxh
        self.geom = self.create_geometry()
        self.mesh = self.generate_mesh()
        self.s_params = []
        self.domains = {"dielectric": {"type": "dielectric", "epsilon": substrate_eps}}
        self.boundaries = {"air_outer": {"type": "outer"}, "ground": {"type": "pec"}, "patch": {"type": "pec"}}
        self.lumped_elements = {"loading_pin": {"type": "feed"}}
    
    def create_geometry(self):
        cm = 1e-2
        mm = 1e-3
        um = 1e-6

        box_y = 32 * mm
        box_x = 45 * mm

        ground_x = 35 * mm

        start_y = 2.5 * mm
        wx = 6 * mm
        wy = 26 * mm

        gw = 7 * mm
        feed_gap = 0.8 * mm
        line_width = 1 * mm
        substrate_height = 1.6 * mm

        air_delta = 36 * mm
        copper_thickness = 35 * um

        antenna_part = (
            MoveTo(ground_x, start_y)
            .Line(wx+line_width)
            .Rotate(90)
            .Line(wy)
            .Rotate(90)
            .Line(line_width)
            .Rotate(90)
            .Line(wy - gw - line_width)
            .Rotate(-90)
            .Line(wx - feed_gap)
            .Rotate(90)
            .Line(line_width)
            .Rotate(90)
            .Line(wx - feed_gap)
            .Rotate(-90)
            .Line(gw - line_width)
            .Rotate(-90)
            .Line(wx)
            .Close()
            .Face()
            .Move((0, 0, substrate_height))
        )
        antenna_part = antenna_part.Extrude(copper_thickness)

        ground_copper = MoveTo(0,0).Rectangle(ground_x, box_y).Face().Move((0, 0, substrate_height))
        ground_copper = ground_copper.Extrude(copper_thickness)

        dielectric = MoveTo(0,0).Rectangle(box_x, box_y).Face().Extrude(substrate_height)

        pins = MoveTo(ground_x, start_y + gw + line_width/2).Rectangle(feed_gap, line_width/2).Face().Move((0, 0, substrate_height))
        pins.edges.Min(Y).name = "loading_pin"

        air_box = Box(
            (-air_delta, -air_delta, -air_delta),
            (air_delta + box_x, air_delta + box_y, air_delta + substrate_height),
        )
        air_box.faces.name = "air_outer"
        air_box = air_box - dielectric
        air_box.faces.col = (0, 0, 1, 0.3)
        air_box.solids.name = "air"
        ground_copper.faces.name = "ground"
        antenna_part.faces.name = "patch"
        dielectric.solids.name = "dielectric"
        dielectric.faces.col = (1, 1, 0)

        shape = Glue([antenna_part, ground_copper, dielectric, pins])

        shape = Glue([shape, air_box])

        geom = OCCGeometry(shape)
        ngsolve.Draw(geom)
        return geom

    def generate_mesh(self):
        ngmesh = self.geom.GenerateMesh(maxh=0.01, segmentsperedge=0)
        mesh = ngsolve.Mesh(ngmesh)
        ngsolve.Draw(mesh)
        return mesh

class PatchAntennaCase:
    def __init__(
        self,
        pw=49.4e-3, 
        pl=41.3e-3, 
        eps=2.2, 
        maxh=0.03
    ):
        self.pw = pw
        self.pl = pl
        self.eps = eps
        self.maxh = maxh
        self.geom = self.create_geometry()
        self.mesh = self.generate_mesh()
        self.domains = {"dielectric": {"type": "dielectric", "epsilon": eps}}
        self.boundaries = {"air_outer": {"type": "outer"}, "ground": {"type": "pec"}, "patch": {"type": "pec"}}
        self.lumped_elements = {"loading_pin": {"type": "feed"}}

    
    def create_geometry(self):
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

    def generate_mesh(self):
        ngmesh = self.geom.GenerateMesh(maxh=0.03, segmentsperedge=0)
        mesh = ngsolve.Mesh(ngmesh)
        return mesh