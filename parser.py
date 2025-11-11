import os

import re

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
class OrbitalData:
    no:        list[int]
    energy:    NDArray[np.float64]
    occupancy: NDArray[np.float64]

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
    
    def get_orbital_energies(self, Eh: bool = False) -> OrbitalData:
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

        return OrbitalData(
            no=no,
            energy=np.array(energy, dtype=np.float64),
            occupancy=np.array(occupancy, dtype=np.float64)
        )

    def get_excited_states(self, soc_corrected: bool = False) -> ExcitedStateData:
        EXCITED_STATE_LINE_RE = re.compile(
            r'STATE\s+\d+:\s+E=\s+\d+\.\d+\s+au\s+\d+\.\d+\s+eV'
            )
        EXCITED_STATE_RE  = re.compile(r'STATE\s+(\d+):')
        ENERGY_RE = re.compile(r'(\d+\.\d+)\s+eV')
        MULT_RE   = re.compile(r'Mult\s+(\d)')

        state  = []
        energy = []
        mult   = []
        
        excited_states_indices = []
        for i in range(len(self.raw)):
            line = self.raw[i]
            if re.match(EXCITED_STATE_LINE_RE, line):
                state_match = re.search(EXCITED_STATE_RE, line)
                if state_match:
                    state_val = int(state_match.group(1))
                    state.append(state_val)

                energy_match = re.search(ENERGY_RE, line)
                if energy_match:
                    energy_val = float(energy_match.group(1))
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
    

if __name__ == "__main__":
    parser = OrcaParser("tests/xanes.out")

    xanes = parser.get_absorption_spectrum()
    orbitals = parser.get_orbital_energies()
    excited_states = parser.get_excited_states()

    # print(orbitals.energy)
    # print(xanes.transition)
    for state, energy in zip(excited_states.state, excited_states.energy):
        print(f"{state: 5d} {energy: 10.3f}")

    for n, energy in zip(orbitals.no, orbitals.energy):
        print(f"{n: 3d} {energy: 10.3f}")
