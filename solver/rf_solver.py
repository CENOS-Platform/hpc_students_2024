import math
import ngsolve
import time

class SolverRF:
    def __init__(self, mesh, domains, boundaries, lumped_elements, properties):
        self.mesh = mesh
        self.domains = domains
        self.boundaries = boundaries
        self.lumped_elements = lumped_elements
        self.omega = properties["frequency"] * 2 * math.pi
        self.properties = properties
        self.matrix = None
        self.load_vector = None
        self.electric_field = None
        self.fes = None
        self.pec_boundaries = self.mesh.Boundaries("|".join([name for name, val in self.boundaries.items() if val["type"] == "pec"]))
        self.outer = self.mesh.Boundaries("|".join([name for name, val in self.boundaries.items() if val["type"] == "outer"]))
        self.feed_line = self.mesh.BBoundaries("|".join([name for name, val in self.lumped_elements.items() if val["type"] == "feed"]))
        self.__assign_materials()
        self.load_type = 1

    def __assign_materials(self):

        self.mur = self.mesh.MaterialCF({mat: val.get("mur", 1) for mat, val in self.domains.items()}, default=1)
        mu0 = 1.257e-6
        
        self.epsilonr = self.mesh.MaterialCF({mat: val.get("epsilon", 1) for mat, val in self.domains.items()}, default=1)
        epsilon0 = 8.854188e-12

        self.k0 = self.omega * ngsolve.sqrt(epsilon0 * mu0)
        self.Z0 = ngsolve.sqrt(mu0/epsilon0)


    def assemble(self):
        self.fes = ngsolve.HCurl(self.mesh, order = 1, dirichlet = self.pec_boundaries, complex=True, autoupdate=True)
        u = self.fes.TrialFunction()
        v = self.fes.TestFunction()

        self.matrix = ngsolve.BilinearForm(self.fes)
        dline = ngsolve.comp.DifferentialSymbol(ngsolve.BBND)
        self.load_vector = ngsolve.LinearForm(self.fes, autoupdate=True)
        tangent  = ngsolve.specialcf.tangential(self.mesh.dim)
        
        # on all domains
        self.matrix += 1/self.mur * ngsolve.curl(u) * ngsolve.curl(v) * ngsolve.dx
        self.matrix += - self.k0**2 * self.epsilonr * u * v * ngsolve.dx

        # outer boundary Sommerfeld radiation condition
        self.matrix += 1j * self.k0 * u.Trace() * v.Trace() * ngsolve.ds(self.outer)

        # on feed line
        self.load_vector +=1j * self.k0 * self.Z0 * tangent * 1.0/50 * v.Trace().Trace() * dline(self.feed_line)

        self.matrix.Assemble()
        self.load_vector.Assemble()
        self.electric_field = ngsolve.GridFunction(self.fes, autoupdate=True)

    def solve(self):
        start = time.perf_counter()

        print("SOLVE DOFS", self.fes.ndof)
        with ngsolve.TaskManager():
            inv = self.matrix.mat.Inverse(self.fes.FreeDofs(), inverse="pardiso")
            self.electric_field.vec.data = inv * self.load_vector.vec

        end = time.perf_counter()
        print(f"Calculation took {end - start:0.4f} seconds")

    def finalize(self):
        efield = self.electric_field
        voltage = ngsolve.Integrate( efield * ngsolve.specialcf.tangential(self.mesh.dim) , self.feed_line, ngsolve.BBND)
        gamma = ((voltage - 1) / (voltage + 1))
        s11 = 20 * math.log10(abs(gamma))
        print("S11", s11)
        print ("voltage:", voltage)
        # uncomment to print out the electric field
        # vtk = ngsolve.VTKOutput(self.mesh,coefs=[efield.real, efield.imag],
        #         names=["efieldre", "efieldim"],
        #         filename="D:/test/vtk_e_field", legacy=True,
        #         subdivision = 0)
        # vtk.Do()
        return s11
    
    