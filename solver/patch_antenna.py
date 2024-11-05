from netgen.occ import MoveTo, Box, Glue, OCCGeometry, X, Y, Z
# It seems that the import of netgen.gui is necessary for the GUI to start
#import netgen.gui # pylint: disable=unused-import
import ngsolve
import matplotlib.pyplot as plt
import numpy as np
from rf_solver import SolverRF

class PatchAntennaCase:
    def __init__(self):
        self.geom = self.create_geometry()
        self.mesh = self.generate_mesh()
        self.frequencies = np.arange(2.1e9, 2.85e9, 0.05e9)
        # reference s11 calculated for np.arange(2.1e9, 2.85e9, 0.05e9) frequencies with cenos
        # calculation was done for approx 600 k DOF's (70 k elements with second order finite element space)
        self.reference_s11 = [ -0.172449,-0.247049, -0.384477, -0.682518, -1.52705, -5.43741,
                            -10.83041,-2.13574, -0.801105, -0.408482, -0.248971, -0.171112,
                            -0.128651, -0.103881, -0.088895]
        self.s_params = []
        self.domains = {"dielectric": {"type": "dielectric", "epsilon": 2.2}}
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
    
    def calculate(self, refinement_function = None):
        self.s_params = []
        for freq in self.frequencies:
            solver = SolverRF(self.mesh, self.domains, self.boundaries, self.lumped_elements, {"frequency": freq})
            solver.assemble()
            solver.solve()
            solver.finalize()
            self.s_params.append(solver.s11)
            # we set refinement flags for next call to refine after each frequency
            if refinement_function is not None:
                refinement_function(self.mesh, solver.electric_field)
    
    def refine_mesh(self):
        self.mesh.Refine()

if __name__ == "__main__":
    patch_antenna_case = PatchAntennaCase()
    #ngsolve.Draw(patch_antenna_case.mesh)

    # example - simple function for refinement. This is what we want to make good!
    def refinement_func(mesh, electric_field):
        """Simple function defined to refine elements with high electric field"""
        solver_fes = electric_field.space
        max_dof = np.max(solver.electric_field.vec)
        for el in mesh.Elements(ngsolve.VOL):
            for edge in el.edges:
                for dof in solver_fes.GetDofNrs(edge):
                    if electric_field.vec[dof] > 0.5 * max_dof:
                        mesh.SetRefinementFlag(el, True)

    # now let's use the function in the calculate method
    patch_antenna_case.calculate()
    s_params_1 = patch_antenna_case.s_params
    residuals_1 = np.array(s_params_1) - np.array(patch_antenna_case.reference_s11)
    norm_residuals_1 = np.linalg.norm(residuals_1)
    print("Residuals after first calculation: ", norm_residuals_1)
    patch_antenna_case.refine_mesh()


    patch_antenna_case.calculate()
    s_params_2 = patch_antenna_case.s_params
    residuals_2 = np.array(s_params_2) - np.array(patch_antenna_case.reference_s11)
    norm_residuals_2 = np.linalg.norm(residuals_2)
    print("Residuals after second calculation: ", norm_residuals_2)
    patch_antenna_case.refine_mesh()

    patch_antenna_case.calculate()
    s_params_3 = patch_antenna_case.s_params
    residuals_3 = np.array(s_params_3) - np.array(patch_antenna_case.reference_s11)
    norm_residuals_3 = np.linalg.norm(residuals_3)
    print("Residuals after third calculation: ", norm_residuals_3)
    patch_antenna_case.refine_mesh()

    patch_antenna_case.calculate()
    s_params_4 = patch_antenna_case.s_params
    residuals_4 = np.array(s_params_4) - np.array(patch_antenna_case.reference_s11)
    norm_residuals_4 = np.linalg.norm(residuals_4)
    print("Residuals after fourth calculation: ", norm_residuals_4)


    # the calculation time quickly explodes from sub-second for each frequency to at the beginning
    # to more than 120 seconds per frequency after last refinement. This may different significantly on different PCs.
    # It is clear that this method does not scale well
    # and further refinement is not possible. We need to find a better way to mark mesh elements for refinement

    plt.figure()
    plt.plot(patch_antenna_case.frequencies, s_params_1)
    plt.plot(patch_antenna_case.frequencies, s_params_2)
    plt.plot(patch_antenna_case.frequencies, s_params_3)
    plt.plot(patch_antenna_case.frequencies, s_params_4)
    plt.plot(patch_antenna_case.frequencies, patch_antenna_case.reference_s11, linewidth=2.5)
    plt.show()
