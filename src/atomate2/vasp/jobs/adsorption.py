"""Jobs used in the calculation of surface adsorption energy."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, Molecule, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.schemas.adsorption import AdsorptionDocument

# from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class BulkRelaxMaker(BaseVaspMaker):
    """Maker for molecule relaxation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the relaxation calculation.
    """

    name: str = "bulk_relax_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings={"reciprocal_density": 60},
            user_incar_settings={
                "ISIF": 3,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LDAU": False,
                "NSW": 300,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )


@dataclass
class MolRelaxMaker(BaseVaspMaker):
    """Maker for molecule relaxation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the relaxation calculation.
    """

    name: str = "mol_relax_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[1, 1, 1]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
            user_incar_settings={
                "ISIF": 2,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LDAU": False,
                "NSW": 300,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )


@dataclass
class MolStaticMaker(BaseVaspMaker):
    """Maker for molecule static energy calculation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the static energy calculation.
    """

    name: str = "mol_static_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[1, 1, 1]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
            user_incar_settings={
                "ENCUT": 700,
                "IBRION": -1,
                "GGA": "RP",
                "EDIFF": 1e-7,
                "LDAU": False,
                "NSW": 0,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )


@dataclass
class SlabRelaxMaker(BaseVaspMaker):
    """Maker for adsorption slab relaxation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the relaxation calculation.
    """

    name: str = "slab_relax_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings={"reciprocal_density": 60},
            user_incar_settings={
                "ISIF": 2,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LDAU": False,
                "NSW": 300,
                "NELM": 500,
            },
            auto_ispin=False,
        )
    )


@dataclass
class SlabStaticMaker(BaseVaspMaker):
    """Maker for slab static energy calculation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the static energy calculation.
    """

    name: str = "slab_static_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings={"reciprocal_density": 100},
            user_incar_settings={
                "ENCUT": 700,
                "IBRION": -1,
                "GGA": "RP",
                "EDIFF": 1e-7,
                "LDAU": False,
                "NSW": 0,
                "NELM": 500,
            },
            auto_ispin=False,
        )
    )


def get_boxed_molecule(molecule: Molecule) -> Structure:
    """Get the molecule structure.

    Parameters
    ----------
    molecule: Molecule
        The molecule to be adsorbed.

    Returns
    -------
    Structure
        The molecule structure.
    """
    return molecule.get_boxed_structure(10, 10, 10)


def remove_adsorbate(slab: Structure) -> Structure:
    """
    Remove adsorbate from the given slab.

    Parameters
    ----------
    slab: Structure
        A pymatgen Slab object, potentially with adsorbates.

    Returns
    -------
    Structure
        The modified slab with adsorbates removed.
    """
    adsorbate_indices = []
    for i, site in enumerate(slab):
        if site.properties.get("surface_properties") == "adsorbate":
            adsorbate_indices.append(i)
    # Reverse the indices list to avoid index shifting after removing sites
    adsorbate_indices.reverse()
    # Remove the adsorbate sites
    for idx in adsorbate_indices:
        slab.remove_sites([idx])
    return slab


@job
def generate_slab(
    bulk_structure: Structure,
    min_slab_size: float,
    surface_idx: tuple,
    min_vacuum_size: float,
    min_lw: float,
) -> Structure:
    """Generate the adsorption slabs without adsorbates.

    Parameters
    ----------
    bulk_structure: Structure
        The bulk/unit cell structure.
    min_slab_size: float
        The minimum size of the slab. In Angstroms or number of hkl planes.
        See pymatgen.core.surface.SlabGenerator for more details.
    surface_idx: tuple
        The Miller index [h, k, l] of the surface.
    min_vacuum_size: float
        The minimum size of the vacuum region. In Angstroms or number of hkl planes.
        See pymatgen.core.surface.SlabGenerator for more details.
    min_lw: float
        The minimum length and width of the slab.

    Returns
    -------
    Structure
        The slab structure without adsorbates.
    """
    hydrogen = Molecule([Element("H")], [[0, 0, 0]])
    slab_generator = SlabGenerator(
        bulk_structure,
        surface_idx,
        min_slab_size,
        min_vacuum_size,
        primitive=False,
        center_slab=True,
    )
    temp_slab = slab_generator.get_slab()
    ads_slabs = AdsorbateSiteFinder(temp_slab).generate_adsorption_structures(
        hydrogen, translate=True, min_lw=min_lw
    )
    return remove_adsorbate(ads_slabs[0])


@job
def generate_adslabs(
    bulk_structure: Structure,
    molecule_structure: Molecule,
    min_slab_size: float,
    surface_idx: tuple,
    min_vacuum_size: float,
    min_lw: float,
) -> list[Structure]:
    """Generate the adsorption slabs with adsorbates.

    Parameters
    ----------
    bulk_structure: Structure
        The bulk/unit cell structure.
    molecule_structure: Structure
        The molecule to be adsorbed.
    min_slab_size: float
        The minimum size of the slab. In Angstroms or number of hkl planes.
    surface_idx: tuple
        The Miller index [h, k, l] of the surface.
    min_vacuum_size: float
        The minimum size of the vacuum region. In Angstroms or number of hkl planes.
    min_lw: float
        The minimum length and width of the slab.

    Returns
    -------
    list[Structure]
        The list of all possible configurations of slab structures with adsorbates.
    """
    slab_generator = SlabGenerator(
        bulk_structure,
        surface_idx,
        min_slab_size,
        min_vacuum_size,
        primitive=False,
        center_slab=True,
    )
    slab = slab_generator.get_slab()

    return AdsorbateSiteFinder(slab).generate_adsorption_structures(
        molecule_structure, translate=True, min_lw=min_lw
    )


@job
def run_adslabs_job(
    adslab_structures: list[Structure],
    relax_maker: SlabRelaxMaker,
    static_maker: SlabStaticMaker,
) -> Response:
    """Workflow of running the adsorption slab calculations.

    Parameters
    ----------
    adslab_structures: list[Structure]
        The list of all possible configurations of slab structures with adsorbates.
    relax_maker: AdslabRelaxMaker
        The relaxation maker for the adsorption slab structures.
    static_maker: SlabStaticMaker
        The static maker for the adsorption slab structures.

    Returns
    -------
    Flow
        The flow of the adsorption slab calculations.
    """
    adsorption_jobs = []
    ads_outputs = defaultdict(list)

    for i, ad_structure in enumerate(adslab_structures):
        ads_job = relax_maker.make(structure=ad_structure, prev_dir=None)
        ads_job.append_name(f"adsconfig_{i}")

        adsorption_jobs.append(ads_job)
        ads_outputs["configuration_number"].append(i)
        ads_outputs["relaxed_structures"].append(ads_job.output.structure)

        prev_dir_ads = ads_job.output.dir_name

        static_job = static_maker.make(
            structure=ads_job.output.structure, prev_dir=prev_dir_ads
        )
        static_job.append_name(f"static_adsconfig_{i}")
        adsorption_jobs.append(static_job)

        ads_outputs["static_energy"].append(static_job.output.output.energy)
        ads_outputs["dirs"].append(ads_job.output.dir_name)

    ads_flow = Flow(adsorption_jobs, ads_outputs)
    return Response(replace=ads_flow)


@job(output_schema=AdsorptionDocument)
def adsorption_calculations(
    adslab_structures: list[Structure],
    adslabs_data: dict[str, list],
    molecule_dft_energy: float,
    slab_dft_energy: float,
) -> AdsorptionDocument:
    """Calculate the adsorption energies and return an AdsorptionDocument instance.

    Parameters
    ----------
    adslab_structures : list[Structure]
        The list of all possible configurations of slab structures with adsorbates.
    adslabs_data : dict[str, list]
        Dictionary containing static energies and directories of the adsorption slabs.
    molecule_dft_energy : float
        The static energy of the molecule.
    slab_dft_energy : float
        The static energy of the slab.

    Returns
    -------
    AdsorptionDocument
        An AdsorptionDocument instance containing all adsorption data.
    """
    adsorption_energies = []
    configuration_numbers = []
    job_dirs = []

    for idx in range(len(adslab_structures)):
        ads_energy = (
            adslabs_data["static_energy"][idx] - molecule_dft_energy - slab_dft_energy
        )
        adsorption_energies.append(ads_energy)
        configuration_numbers.append(idx)
        job_dirs.append(adslabs_data["dirs"][idx])

    # Sort the data by adsorption energy
    sorted_indices = sorted(
        range(len(adsorption_energies)), key=lambda k: adsorption_energies[k]
    )

    # Apply the sorted indices to all lists
    sorted_structures = [adslab_structures[i] for i in sorted_indices]
    sorted_configuration_numbers = [configuration_numbers[i] for i in sorted_indices]
    sorted_adsorption_energies = [adsorption_energies[i] for i in sorted_indices]
    sorted_job_dirs = [job_dirs[i] for i in sorted_indices]

    # Create and return the AdsorptionDocument instance
    return AdsorptionDocument(
        structures=sorted_structures,
        configuration_numbers=sorted_configuration_numbers,
        adsorption_energies=sorted_adsorption_energies,
        job_dirs=sorted_job_dirs,
    )


class AdsorptionFailureChecker:
    def __init__(
        self,
        init_structure: Structure,
        final_structure: Structure,
        atoms_tag: list[int],
        final_slab_structure: Structure | None = None,
        surface_change_cutoff_multiplier: float = 1.5,
        desorption_cutoff_multiplier: float = 1.5,
    ):
        """
        Flag anomalies based on initial and final structure of a adsorption relaxation.

        Args:
            init_structure (Structure): the adslab in its initial state
            final_structure (Structure): the adslab in its final state
            atoms_tag (List[int]): the atom tags; 0=bulk, 1=surface, 2=adsorbate
            final_slab_structure (Structure, optional): the relaxed slab. If unspecified, this defaults
                to using the initial adslab without the adsorbate.
            surface_change_cutoff_multiplier (float, optional): cushion for small atom movements
                when assessing atom connectivity for reconstruction
            desorption_cutoff_multiplier (float, optional): cushion for physisorbed systems to not
                be discarded. Applied to the covalent radii.
        """
        self.init_structure = init_structure
        self.final_structure = final_structure
        self.atoms_tag = atoms_tag
        self.surface_change_cutoff_multiplier = surface_change_cutoff_multiplier
        self.desorption_cutoff_multiplier = desorption_cutoff_multiplier

        if final_slab_structure is None:
            slab_indices = [i for i, tag in enumerate(self.atoms_tag) if tag != 2]
            slab_sites = [self.init_structure[i] for i in slab_indices]
            self.final_slab_structure = Structure.from_sites(slab_sites)
        else:
            self.final_slab_structure = final_slab_structure

    def _get_connectivity(
        self, structure: Structure, cutoff_multiplier: float = 1.0
    ) -> np.ndarray:
        """
        Generate the connectivity of a structure by computing pairwise distances.

        Args:
            structure (Structure): Pymatgen Structure object
            cutoff_multiplier (float, optional): Cushion for small atom movements when assessing
                atom connectivity.

        Returns
        -------
            np.ndarray: The connectivity matrix of the structure.
        """
        num_sites = len(structure)
        adjacency_matrix = np.zeros((num_sites, num_sites), dtype=int)

        # Get covalent radii for each site
        covalent_radii = []
        for site in structure.sites:
            element = str(site.specie)
            covalent_radius = CovalentRadius.radius[element]
            if covalent_radius is None:
                raise ValueError(f"Covalent radius not defined for element {element}")
            covalent_radii.append(covalent_radius)

        # Compute adjacency matrix
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                cutoff = (covalent_radii[i] + covalent_radii[j]) * cutoff_multiplier
                distance = structure.get_distance(i, j)
                if distance <= cutoff:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1

        return adjacency_matrix

    def is_adsorbate_dissociated(self) -> bool:
        """
        Tests if the initial adsorbate connectivity is maintained.

        Returns
        -------
            bool: True if the connectivity was not maintained, otherwise False
        """
        adsorbate_indices = [i for i, tag in enumerate(self.atoms_tag) if tag == 2]
        init_adsorbate_structure = Structure.from_sites(
            [self.init_structure[i] for i in adsorbate_indices]
        )
        final_adsorbate_structure = Structure.from_sites(
            [self.final_structure[i] for i in adsorbate_indices]
        )

        init_connectivity = self._get_connectivity(init_adsorbate_structure)
        final_connectivity = self._get_connectivity(final_adsorbate_structure)

        return not np.array_equal(init_connectivity, final_connectivity)

    def is_surface_changed(self) -> bool:
        """
        Tests bond breaking/forming events within a tolerance on the surface so
        that systems with significant adsorbate-induced surface changes may be discarded
        since the reference to the relaxed slab may no longer be valid.

        Returns
        -------
            bool: True if the surface is reconstructed, otherwise False
        """
        surface_indices = [i for i, tag in enumerate(self.atoms_tag) if tag != 2]
        final_surface_structure = Structure.from_sites(
            [self.final_structure[i] for i in surface_indices]
        )

        adslab_connectivity = self._get_connectivity(final_surface_structure)
        slab_connectivity_w_cushion = self._get_connectivity(
            self.final_slab_structure, self.surface_change_cutoff_multiplier
        )
        ### surface reconstruction or aggregation
        slab_test = np.any(adslab_connectivity - slab_connectivity_w_cushion == 1)

        adslab_connectivity_w_cushion = self._get_connectivity(
            final_surface_structure, self.surface_change_cutoff_multiplier
        )
        slab_connectivity = self._get_connectivity(self.final_slab_structure)
        ### bond breaking or surface degradation
        adslab_test = np.any(slab_connectivity - adslab_connectivity_w_cushion == 1)

        return slab_test or adslab_test

    def is_adsorbate_desorbed(self) -> bool:
        """
        Checks if the adsorbate binding atoms have no connection with slab atoms,
        considering it desorbed.

        Returns
        -------
            bool: True if there is desorption, otherwise False
        """
        adsorbate_indices = [i for i, tag in enumerate(self.atoms_tag) if tag == 2]
        surface_indices = [i for i, tag in enumerate(self.atoms_tag) if tag != 2]
        final_connectivity = self._get_connectivity(
            self.final_structure, self.desorption_cutoff_multiplier
        )
        ### The adsorbate is considered desorbed if it has no bonds with the surface atoms in the final structure.
        for idx in adsorbate_indices:
            if np.sum(final_connectivity[idx][surface_indices]) >= 1:
                return False
        return True

    def is_adsorbate_intercalated(self) -> bool:
        """
        Ensures the adsorbate isn't interacting with an atom that is not allowed to relax.

        Returns
        -------
            bool: True if any adsorbate atom neighbors a frozen atom, otherwise False
        """
        adsorbate_indices = [i for i, tag in enumerate(self.atoms_tag) if tag == 2]
        frozen_atoms_indices = [i for i, tag in enumerate(self.atoms_tag) if tag == 0]
        final_connectivity = self._get_connectivity(self.final_structure)

        for idx in adsorbate_indices:
            if np.sum(final_connectivity[idx][frozen_atoms_indices]) >= 1:
                return True
        return False


def structure_matcher(structure1, structure2):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    # Initialize the StructureMatcher
    matcher = StructureMatcher(
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        allow_subset=False,
        stol=0.01,  # Site tolerance in fractional coordinates
        angle_tol=5,  # Angle tolerance in degrees
    )

    # Compare the structures
    are_structures_equal = matcher.fit(structure1, structure2)
    return are_structures_equal


def check_dissociation(from_vasp, structure_orig, vasp_orig, vasp_relaxed, tags):
    if from_vasp:
        if structure_matcher(structure_orig, vasp_orig):
            # Transfer the tags to CONTCAR_from_file
            CONTCAR_from_file.add_site_property("surface_properties", tags)

            anomaly_detector = AdsorptionFailureChecker(
                init_structure=structure_orig,
                final_structure=vasp_relaxed,
                atoms_tag=[
                    0 if tag == "subsurface" else 1 if tag == "surface" else 2
                    for tag in tags
                ],
            )

            # Use the methods of anomaly_detector
            dissociated = anomaly_detector.is_adsorbate_dissociated()
            surface_changed = anomaly_detector.is_surface_changed()
            desorbed = anomaly_detector.is_adsorbate_desorbed()
            intercalated = anomaly_detector.is_adsorbate_intercalated()

            # Print the results
            print(f"Adsorbate Dissociated: {dissociated}")
            print(f"Surface Changed: {surface_changed}")
            print(f"Adsorbate Desorbed: {desorbed}")
            print(f"Adsorbate Intercalated: {intercalated}")
        else:
            print("The structures do not match.")
    else:
        anomaly_detector = AdsorptionFailureChecker(
            init_structure=structure_orig,
            final_structure=vasp_relaxed,
            atoms_tag=[
                0 if tag == "subsurface" else 1 if tag == "surface" else 2
                for tag in tags
            ],
        )
        # Use the methods of anomaly_detector
        dissociated = anomaly_detector.is_adsorbate_dissociated()
        surface_changed = anomaly_detector.is_surface_changed()
        desorbed = anomaly_detector.is_adsorbate_desorbed()
        intercalated = anomaly_detector.is_adsorbate_intercalated()

        # Print the results
        print(f"Adsorbate Dissociated: {dissociated}")
        print(f"Surface Changed: {surface_changed}")
        print(f"Adsorbate Desorbed: {desorbed}")
        print(f"Adsorbate Intercalated: {intercalated}")
