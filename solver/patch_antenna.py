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
        # reset refinement
        for el in self.mesh.Elements(ngsolve.VOL):
            self.mesh.SetRefinementFlag(el, False)

        self.s_params = []
        for freq in self.frequencies:
            solver = SolverRF(self.mesh, self.domains, self.boundaries, self.lumped_elements, {"frequency": freq})
            solver.assemble()
            solver.solve()
            solver.finalize()
            self.s_params.append(solver.s11)
            # we set refinement flags for next call to refine after each frequency
            if refinement_function is not None:
                refinement_function(self.mesh, solver)
    
    def refine_mesh(self):
        self.mesh.Refine()

if __name__ == "__main__":
    # create instance of patch antenna case
    patch_antenna_case = PatchAntennaCase()
    #ngsolve.Draw(patch_antenna_case.mesh)

    # example - simple function for refinement. This is what we want to make good!
    def refinement_func(mesh, solver):
        """Simple function defined to refine elements with high electric field"""
        solver_fes = solver.electric_field.space
        max_dof = np.max(solver.electric_field.vec)
        for el in mesh.Elements(ngsolve.VOL):
            for edge in el.edges:
                for dof in solver_fes.GetDofNrs(edge):
                    if solver.electric_field.vec[dof] > 0.8 * max_dof:
                        mesh.SetRefinementFlag(el, True)
    
    def refinement_function2(mesh, solver):
        errors = solver.estimate_error()
        max_error = np.max(errors)
        for idx, el in enumerate(mesh.Elements(ngsolve.VOL)):
            if errors[idx] > 0.8 * max_error:
                mesh.SetRefinementFlag(el, True)


    # now let's use the function in the calculate method
    num_iterations = 6
    s_params_list = []
    norm_residuals_list = []

    # here we loop over iterations and refine mesh each time.
    # we also pass our refinement function to calculate method to set refinement flags for mesh elements
    for i in range(num_iterations):
        patch_antenna_case.calculate(refinement_function2)
        s_params = patch_antenna_case.s_params
        s_params_list.append(s_params)
        residuals = np.array(s_params) - np.array(patch_antenna_case.reference_s11)
        norm_residuals = np.linalg.norm(residuals)
        norm_residuals_list.append(norm_residuals)
        print(f"Residuals after calculation {i + 1}: ", norm_residuals)
        patch_antenna_case.refine_mesh()

    # Plotting the results
    plt.figure()
    for i, s_params in enumerate(s_params_list):
        plt.plot(patch_antenna_case.frequencies, s_params, label=f"Iteration {i}")
    plt.plot(patch_antenna_case.frequencies, patch_antenna_case.reference_s11, linewidth=2.5, label="Reference")
    plt.legend()
    plt.show()
