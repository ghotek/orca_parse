import os

import re

from dataclasses import dataclass

import numpy as np

from numpy.typing import NDArray
from typing import Any


def read_orca_output(path: str | os.PathLike) -> list[str]:
    try:
        with open(path, "r") as orca_output:
            text = orca_output.readlines()
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"ORCA output file not found at {path}")
    except Exception as e:
        raise RuntimeError(f"Error reading ORCA output: {e}")
        

@dataclass
class SpectrumData:
    energy:     NDArray[np.float64]
    absorption: NDArray[np.float64]
    transition: list[str]

@dataclass
class MolecularOrbitalData:
    no:        list[int] = None
    energy:    NDArray[np.float64] = None
    occupancy: NDArray[np.float64] = None

    ao_coefficients: NDArray[np.float64] = None
    ao_labels      : list[str]           = None

    def print_top_ao2mo(
            self, from_mo: int, to_mo: int,
            atom_labels: list[str], ao_thr: float = 0.1
        ) -> None:
        MO_INFO_TEMPLATE =     "MO {0:3s} {1:14.10}eV:"
        POS_HTOP_AO_TEMPLATE = "+   {0:4s}  {1}"
        POS_TOP_AO_TEMPLATE  = "    {0:4s}  {1}"
        NEG_HTOP_AO_TEMPLATE = "-   {0:4s}  {1}"
        NEG_TOP_AO_TEMPLATE  = "    {0:4s}  {1}"
    
        AO_COEFFS_TEMPLATE = "{0:6s} ({1:6.3f})"

        MOs = [int(i) for i in range(from_mo, to_mo)]
        for mo in MOs:
            print(MO_INFO_TEMPLATE.format(str(mo), self.energy[mo]))
            sorted_indices = np.argsort(np.abs(self.ao_coefficients[mo]))[::-1]
            
            sdict_ = {
                "positive": {},
                "negative": {}
            }
            for atom_label in atom_labels:
                sdict_["positive"][atom_label] = ""
                sdict_["negative"][atom_label] = ""

            for ao_i in sorted_indices:
                if not np.greater_equal(np.abs(self.ao_coefficients[mo][ao_i]), ao_thr):
                    continue
                
                if self.ao_coefficients[mo][ao_i] > 0.0:
                    key = "positive"
                else:
                    key = "negative"
                
                for atom_label in atom_labels:
                    if atom_label in self.ao_labels[ao_i]:
                        name = atom_label
                        rp_ = atom_label + '_'
                    else:
                        continue

                    sdict_[key][name] += AO_COEFFS_TEMPLATE.format(
                        self.ao_labels[ao_i].replace(rp_, ''), self.ao_coefficients[mo][ao_i]
                        ) + 5 * ' '

            for atom_label_i in range(len(atom_labels)):
                atom_label = atom_labels[atom_label_i]
                if atom_label_i == 0:
                    print(POS_HTOP_AO_TEMPLATE.format(
                        atom_label, sdict_["positive"][atom_label])
                        )
                    continue

                print(POS_TOP_AO_TEMPLATE.format(
                    atom_label, sdict_["positive"][atom_label])
                    )
            print()

            for atom_label_i in range(len(atom_labels)):
                atom_label = atom_labels[atom_label_i]
                if atom_label_i == 0:
                    print(NEG_HTOP_AO_TEMPLATE.format(
                        atom_label, sdict_["negative"][atom_label])
                        )
                    continue

                print(NEG_TOP_AO_TEMPLATE.format(
                    atom_label, sdict_["negative"][atom_label])
                    )
            print('\n')
              

@dataclass
class ExcitedStateData:
    state:       list[int]
    energy:      NDArray[np.float64]
    mult:        list[int]
    transitions: list[list[tuple[int, int]]]
    weights:     list[list[float]]


class OrcaParser:
    def __init__(self, path: str | os.PathLike):
        self.raw = read_orca_output(path=path)

    def get_absorption_spectrum(
            self, soc_corrected: bool = False,
            transition_velocity: bool = False,
            cd: bool = False
        ) -> SpectrumData:
        soc_corrected_key       = "SOC CORRECTED"
        transition_velocity_key = "TRANSITION VELOCITY"
        keys = [soc_corrected_key, transition_velocity_key]
        keys_exist = [soc_corrected, transition_velocity]

        spectrum_type = "ABSORPTION SPECTRUM"
        if cd:
            spectrum_type = "CD SPECTRUM"

        header_i = 0
        for j, line in enumerate(self.raw):
            if spectrum_type in line:
                keys_check = [key in line for key in keys]
                if keys_check == keys_exist:
                    header_i = j
                    break
        
        spectrum_table = re.split(r'--+', ''.join(self.raw[header_i:]))[2]
        energy = []
        absorption = []
        transition = []
        for line in spectrum_table.splitlines():
            if len(line) == 0:
                continue

            line_ = re.split(r'\s+', line)
            if len(line_[0]) == 0:
                line_ = line_[1:]

            energy.append(line_[3])
            absorption.append(line_[6])
            transition.append(''.join(line_[:3]))


        return SpectrumData(
            energy=np.array(energy, dtype=np.float64),
            absorption=np.array(absorption, dtype=np.float64),
            transition=transition
        )
    
    def get_orbital_energies(self, Eh: bool = False) -> MolecularOrbitalData:
        header_i = 0
        for j, line in enumerate(self.raw):
            if "ORBITAL ENERGIES" in line:
                header_i = j
                break
            
        ORBITAL_ENERGY_LINE_RE = re.compile(
            r'^\s+(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)'
        )

        no = []
        energy = []
        occupancy = []
        for line in self.raw[header_i + 4:]:
            orbital_energy_match = re.match(ORBITAL_ENERGY_LINE_RE, line)
            if not orbital_energy_match:
                break

            no.append(orbital_energy_match.group(1))
            if Eh:
                energy.append(orbital_energy_match.group(3))
            else:
                energy.append(orbital_energy_match.group(4))
            occupancy.append(orbital_energy_match.group(2))

        return MolecularOrbitalData(
            no=no,
            energy=np.array(energy, dtype=np.float64),
            occupancy=np.array(occupancy, dtype=np.float64)
        )

    def get_excited_states(
            self, au: bool = False
            ) -> ExcitedStateData:
        EXCITED_STATE_LINE_RE = re.compile(
            r'^STATE\s+(\d+)(?:\s+[^\:]+)?\s*:\s*E=\s*([+-]?\d+(?:\.\d+)?)\s*au\s*([+-]?\d+(?:\.\d+)?)\s*eV'
            )
        MULT_RE   = re.compile(r'\bMult\s*=?\s*(\d+)\b')

        state  = []
        energy = []
        mult   = []
        
        excited_states_indices = []
        for i in range(len(self.raw)):
            line = self.raw[i]
            excited_state_match = re.match(EXCITED_STATE_LINE_RE, line)
            if excited_state_match:
                state_val = int(excited_state_match.group(1))
                state.append(state_val)

                energy_val = float(excited_state_match.group(3))
                if au:
                    energy_val = float(excited_state_match.group(2))
                energy.append(energy_val)

                mult_match = re.search(MULT_RE, line)
                if mult_match:
                    mult_val = int(mult_match.group(1))
                    mult.append(mult_val)

                excited_states_indices.append(i)

        TRANSITION_LINE_RE = re.compile(
            r'\s+\d+a\s+->\s+\d+a\s+:\s+\d+\.\d+\s+\(c='
            )
        TRANSITION_RE = re.compile(r'\s+(\d+)a\s+->\s+(\d+)a\s+:')
        WEIGHT_RE     = re.compile(r'(\d+\.\d+)\s+\(c=') 

        transitions = []
        weights     = []
        for ex_st_i in excited_states_indices:
            tr_i = ex_st_i + 1
            tr_line = self.raw[tr_i]

            trs = []
            ws  = []
            while re.match(TRANSITION_LINE_RE, tr_line):
                transition_match = re.search(TRANSITION_RE, tr_line)
                if transition_match:
                    transition_tup = (
                        int(transition_match.group(1)), 
                        int(transition_match.group(2))
                    )
                    trs.append(transition_tup)

                weight_match = re.search(WEIGHT_RE, tr_line)
                if weight_match:
                    weight_val = float(weight_match.group(1))
                    ws.append(weight_val)

                tr_i += 1
                tr_line = self.raw[tr_i]

            transitions.append(trs)
            weights.append(ws)

        return ExcitedStateData(
            state=state,
            energy=np.array(energy, dtype=np.float64),
            mult=mult,
            transitions=transitions,
            weights=weights
        )
    
    def get_ao_coefficients(self, Eh: bool = False) -> MolecularOrbitalData:
        orbitals = self.get_orbital_energies(Eh=Eh)
        no = orbitals.no
        energy = orbitals.energy
        occupancy = orbitals.occupancy
        
        MO_NUMBER_RE   = re.compile(r'^\s*\d+(\s+\d+)*\s*$')
        AO_LABEL_RE    = re.compile(r'^\s*\d+[A-Za-z]+\s+\d*[spdfg][\w\d+-]*')
        COEFF_VALUE_RE = re.compile(r'([+-]?\d*\.\d+)')

        ao_labels = []
        ao_coefficients_dict = {}
        current_mo_block = None

        i = 0
        while i < len(self.raw):
            line = self.raw[i].strip()
            if not line:
                i += 1
                continue
            
            if re.match(MO_NUMBER_RE, line):
                mo_numbers = list(map(int, line.split()))
                i += 2
                
                current_mo_block = mo_numbers
                i += 1
                continue
                
            if line.startswith('--------'):
                i += 1
                continue
                
            if (current_mo_block is not None and re.match(AO_LABEL_RE, line)):
                parts = line.split()
                
                coefficients_match = re.findall(COEFF_VALUE_RE, line)
                if len(coefficients_match) == len(current_mo_block):
                    atom_label    = parts[0]
                    orbital_label = parts[1]
                    ao_label = f"{atom_label}_{orbital_label}"
                    
                    try:
                        coefficients = list(map(float, coefficients_match))
                        if ao_label not in ao_labels:
                            ao_labels.append(ao_label)

                        if ao_label not in ao_coefficients_dict:
                            ao_coefficients_dict[ao_label] = {}

                        for mo_num, coeff in zip(current_mo_block, coefficients):
                            ao_coefficients_dict[ao_label][mo_num] = coeff
                    except ValueError:
                        pass
                else:
                    raise ValueError(f"Can't parse this line: {line}")
            
            i += 1
        
        ao_coefficients = np.zeros(
            shape=(len(no), len(ao_coefficients_dict)), dtype=np.float64
            )
        for i in range(ao_coefficients.shape[0]):
            for j in range(ao_coefficients.shape[1]):
                ao_key = list(ao_coefficients_dict.keys())[j]
                no_key = list(ao_coefficients_dict[ao_key].keys())[i]

                ao_coefficients[i][j] = ao_coefficients_dict[ao_key][no_key]
        
        return MolecularOrbitalData(
            no=no, energy=energy, occupancy=occupancy,
            ao_coefficients=ao_coefficients, ao_labels=ao_labels
        )

if __name__ == "__main__":
    spectrum_parser = OrcaParser("tests/xanes.out")
    xanes = spectrum_parser.get_absorption_spectrum(soc_corrected=False)
    energies = spectrum_parser.get_orbital_energies()
    states = spectrum_parser.get_excited_states()

    sp_calc_parser = OrcaParser("tests/single_point_calc.out")
    orbitals = sp_calc_parser.get_ao_coefficients()

    mask = np.argsort(xanes.absorption)[::-1][:10]
    sign_transitions = [xanes.transition[i] for i in mask]

    TRANSITION_RE = r'(\d+)-\d*[.]?\d+A->(\d+)-\d*[.]?\d+A'
    sign_states = []
    for tr in sign_transitions:
        tr_match = re.match(TRANSITION_RE, tr)
        if tr_match:
            sign_states.append(
                (int(tr_match.group(1)), int(tr_match.group(2)))
                )
            
    for tr in sign_states:
        final_state = tr[1]
        final_state_i = states.state.index(final_state)

        for mo_tr in states.transitions[final_state_i]:
            print("Transition from {} MO to {} MO".format(mo_tr[0], mo_tr[1]))
            print("Atomic orbital contributions of {} MO".format(mo_tr[0]))
            orbitals.print_top_ao2mo(
                from_mo=mo_tr[0], to_mo=mo_tr[0] + 1, atom_labels=['0Au', '1S', '3S']
                )
            
            print("Atomic orbital contributions of {} MO".format(mo_tr[1]))
            orbitals.print_top_ao2mo(
                from_mo=mo_tr[1], to_mo=mo_tr[1] + 1, atom_labels=['0Au', '1S', '3S']
                )
            
            print('\n' + '-' * 100 + '\n')