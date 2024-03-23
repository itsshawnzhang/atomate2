"""Flow for calculating surface adsorption energies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from atomate2.vasp.jobs.adsorption import (
    get_boxed_molecule,
    AdslabRelaxMaker,
    SlabStaticMaker,
    MolStaticMaker,
    MoleculeRelaxMaker,
    generate_slab,
    generate_adslabs,
    run_adslabs_job,
    adsorption_calculations
)
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import TightRelaxMaker

if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core.structure import Structure, Molecule
    from atomate2.vasp.jobs.base import BaseVaspMaker

@dataclass
class AdsorptionMaker(Maker):
    """Makes a flow for calculating adsorption energies."""
    name: str = "adsorption"
    # do we need this?
    get_supercell_size_kwargs: dict = field(default_factory=dict)


    bulk_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    mol_relax_maker: BaseVaspMaker = field(default_factory=MoleculeRelaxMaker())

    adslab_relax_maker: BaseVaspMaker = field(default_factory=AdslabRelaxMaker())

    slab_static_energy_maker: BaseVaspMaker = field(default_factory=SlabStaticMaker())

    mol_static_energy_maker: BaseVaspMaker = field(default_factory=MolStaticMaker())


    def make(
            self,
            molecule: Molecule,
            structure: Structure,
            min_vacuum: float = 20.0,
            min_slab_size: float = 10.0,
            min_lw: float = 10.0,
            surface_idx=(0, 0, 1),
            prev_dir_mol: str | Path | None = None,
            prev_dir_bulk: str | Path | None = None
    ) -> Flow:
        """
        Parameters
        ----------
        molecule
        structure: .Structure
            A pymatgen structure object.
        min_vacuum:
        min_slab_size:
        Minimum size in angstroms of layers containing atoms.
        min_lw: minimum length and width of the slab, only used
                if repeat is None
        surface_idx: Miller index of plane parallel to surface.
        prev_dir: str or Path or None
            A previous VASP calculation directory to copy output files from.

        molecule_dft_energy
        slab_dft_energy

        Returns
        -------

        """

        jobs = []

        molecule_structure = get_boxed_molecule(molecule)

        if self.mol_relax_maker is not None:
            mol_optimize_job = self.mol_relax_maker.make(molecule_structure, prev_dir=prev_dir_mol)
            mol_optimize_job.append_name(f"molecule relaxation job")
            jobs.append(mol_optimize_job)

            optimized_molecule = mol_optimize_job.output.structure
            prev_dir = mol_optimize_job.output.dir_name
        else:
            prev_dir = prev_dir_mol
            optimized_molecule = molecule_structure

        mol_static_job = self.mol_static_energy_maker.make(molecule_structure, prev_dir=prev_dir)
        mol_static_job.append_name(f"molecule static job")
        jobs.append(mol_static_job)

        molecule_dft_energy = mol_static_job.output.output.energy

        if self.bulk_relax_maker is not None:
            bulk_optimize_job = self.bulk_relax_maker.make(structure, prev_dir=prev_dir_bulk)
            bulk_optimize_job.append_name(f"bulk relaxation job")
            jobs.append(bulk_optimize_job)

            optimized_bulk = bulk_optimize_job.output.structure
        else:
            optimized_bulk = structure

        generate_slab_structure = generate_slab(
            bulk_structure=optimized_bulk,
            min_slab_size=min_slab_size,
            surface_idx=surface_idx,
            min_vacuum_size=min_vacuum,
            min_lw=min_lw,
        )
        jobs.append(generate_slab_structure)

        generate_adslabs_structures = generate_adslabs(
            bulk_structure=optimized_bulk,
            molecule_structure=optimized_molecule,
            min_slab_size=min_slab_size,
            surface_idx=surface_idx,
            min_vacuum_size=min_vacuum,
            min_lw=min_lw,
        )
        jobs.append(generate_adslabs_structures)

        if self.adslab_relax_maker is None:
            raise ValueError("adslab_relax_maker shouldn't be Null.")

        # slab relaxation without adsoprtion
        slab_optimize_job = self.adslab_relax_maker.make(generate_slab_structure)
        slab_optimize_job.append_name(f"slab relaxation job")
        jobs.append(slab_optimize_job)

        optimized_slab = slab_optimize_job.output.structure
        prev_dir = slab_optimize_job.output.dir_name

        slab_static_job = self.slab_static_energy_maker.make(optimized_slab, prev_dir=prev_dir)
        slab_static_job.append_name(f"slab static job")
        jobs.append(slab_static_job)

        slab_dft_energy = slab_static_job.output.energy

        vasp_adslabs_calcs = run_adslabs_job(
            adslab_structures=generate_adslabs_structures,
            relax_maker=self.adslab_relax_maker,
            static_maker=self.slab_static_energy_maker
        )
        jobs.append(vasp_adslabs_calcs)

        adsorption_calc = adsorption_calculations(
            bulk_structure=optimized_bulk,
            molecule_structure=optimized_molecule,
            surface_idx=surface_idx,
            adslab_structures=generate_adslabs_structures,
            adslabs_data=vasp_adslabs_calcs.output,
            molecule_dft_energy=molecule_dft_energy,
            slab_dft_energy=slab_dft_energy,
        )
        jobs.append(adsorption_calc)

        return Flow(jobs, adsorption_calc.output)
