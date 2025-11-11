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

    ao_contributions: NDArray[np.float64] = None
    ao_labels       : list[str]           = None

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
        
        orbital_table = ''.join(self.raw[header_i:]).split('*')[0]
        no = []
        energy = []
        occupancy = []
        for line in orbital_table.splitlines()[4:]:
            if len(line) == 0:
                continue

            line_ = re.split(r'\s+', line)
            if len(line_[0]) == 0:
                line_ = line_[1:]

            no.append(int(line_[0]))
            if Eh:
                energy.append(float(line_[3]))
            else:
                energy.append(float(line_[2]))
            occupancy.append(float(line_[1]))

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
    
    def parse_single_point_calc(self) -> MolecularOrbitalData:
        orbitals = self.get_orbital_energies()
        no = orbitals.no
        energy = orbitals.energy
        occupancy = orbitals.occupancy
        
        MO_NUMBER_RE = re.compile(r'^\s*\d+(\s+\d+)*\s*$')
        AO_LABEL_RE  = re.compile(r'^\s*\d+[A-Za-z]+\s+\d*[spdfg][\w\d+-]*')

        ao_labels = []

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
                if len(parts) >= 2 + len(current_mo_block):
                    atom_label = parts[0]
                    orbital_label   = parts[1]
                    ao_label = f"{atom_label}_{orbital_label}"
                    
                    try:
                        coefficients = list(map(float, parts[2:2+len(current_mo_block)]))
                        
                        if ao_label not in ao_labels:
                            ao_labels.append(ao_label)
                        
                        # Сохраняем коэффициенты для каждого MO в текущем блоке
                        for mo_num, coeff in zip(current_mo_block, coefficients):
                            ao_data[atom_label][mo_num] = coeff
                    except ValueError:
                        # Пропускаем строки, которые не удается преобразовать
                        pass
            
            i += 1
        
        # Создаем DataFrame
        if not ao_data:
            return pd.DataFrame(), pd.Series(), pd.Series()
        
        # Получаем все номера MO и сортируем их
        all_mo_numbers = sorted(set().union(*[set(ao_data[ao].keys()) for ao in ao_data]))
        
        # Создаем матрицу коэффициентов
        ao_names = ao_data.keys()
        coefficient_matrix = []
        
        for ao in ao_names:
            row = [ao_data[ao].get(mo, 0.0) for mo in all_mo_numbers]
            coefficient_matrix.append(row)
        
        # Создаем DataFrame
        columns = [f'{mo}' for mo in all_mo_numbers]
        df = pd.DataFrame(coefficient_matrix, index=ao_names, columns=columns)
        
        # Создаем Series для энергий и occupancies
        energy_series = pd.Series({f'{mo}': mo_energies.get(mo, np.nan) for mo in all_mo_numbers})
        occupancy_series = pd.Series({f'{mo}': mo_occupancies.get(mo, np.nan) for mo in all_mo_numbers})

        return df, energy_series, occupancy_series

if __name__ == "__main__":
    parser = OrcaParser("tests/xanes.out")

    xanes = parser.get_absorption_spectrum()
    orbitals = parser.get_orbital_energies()
    excited_states = parser.get_excited_states()

    # print(orbitals.energy)
    # print(xanes.transition)
    for state, energy in zip(excited_states.state, excited_states.energy):
        print(f"{state: 5d} {energy: 10.3f}")

    # for n, energy in zip(orbitals.no, orbitals.energy):
    #     print(f"{n: 3d} {energy: 10.3f}")
