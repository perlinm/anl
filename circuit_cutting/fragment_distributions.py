#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from tensorflow_extension import tf_outer_product

from itertools import product as set_product
from functools import reduce
from copy import deepcopy

SIC, IZXY, ZZXY = "SIC", "IZXY", "ZZXY" # define these to protect against typos

def get_bloch_angles(vec):
    '''
    A pure state pointing in the direction of `vec` takes the form
      \cos(\theta/2) |0> + \exp(i\phi) \sin(\theta/2) |1>,
    where \theta and \phi are respectively the polar and azimuthal angles of `vec`.

    Note that we assume vectors are stored in (Z,X,Y) format, such that
    the *first* entry is the component of `vec` along the computational basis axis,
    and the remaining entries are components in the equatorial plane.
    '''
    vec = np.array(vec) / np.linalg.norm(vec)
    theta = np.arccos(vec[0])
    phi = np.arctan2(vec[2],vec[1])
    return theta, phi

# define state vectors for SIC basis
state_vecs_SIC = [ ( +1, +1, +1 ), ( +1, -1, -1 ), ( -1, +1, -1 ), ( -1, -1, +1 ) ]
state_vecs_SIC = { f"SIC-{jj}" : tuple( np.array(vec) / np.linalg.norm(vec) )
                   for jj, vec in enumerate(state_vecs_SIC) }

# define state vectors for ZXY basis
state_vecs_ZXY = { "+Z" : (+1,0,0),
                   "-Z" : (-1,0,0),
                   "+X" : (0,+1,0),
                   "-X" : (0,-1,0),
                   "+Y" : (0,0,+1),
                   "-Y" : (0,0,-1) }

# collect all string-identified state vectors into one dictionary
state_vecs = dict(state_vecs_ZXY, **state_vecs_SIC)

op_basis_SIC = list(state_vecs_SIC.keys())
op_basis_IZXY = [ "I", "+Z", "+X", "+Y" ]
op_basis_ZZXY = [ "-Z", "+Z", "+X", "+Y" ]
op_basis = { SIC : op_basis_SIC,
             IZXY : op_basis_IZXY,
             ZZXY : op_basis_ZZXY }

# general class for storing conditional distributions
# note: class intended for inheritence by other classes, rather than for direct use
class FragmentDistribution:

    # initialize an empty conditional distribution function
    def __init__(self):
        self._data_dict = {}

    # if asked to print, print the data dictionary
    def __repr__(self):
        return self._data_dict.__repr__()

    # iterate over conditions / distributions
    def __iter__(self):
        return iter(self._data_dict.items())

    # iterate over all sets of conditions
    def all_conditions(self):
        return self._data_dict.keys()

    # iterate over all distributions
    def all_distributions(self):
        return self._data_dict.values()

    def _combine_conds(self, init_conds, exit_conds):
        init_conds = ( cond if len(cond) == 3 else (True,) + cond
                       for cond in init_conds )
        exit_conds = ( cond if len(cond) == 3 else (False,) + cond
                       for cond in exit_conds )
        return frozenset().union(init_conds, exit_conds)

    # add data to the conditional distribution
    def add(self, *args):
        assert( len(args) in [ 2, 3 ] )

        if len(args) == 3:
            conditions = self._combine_conds(*args[:2])
        else: # len(args) == 1:
            conditions = frozenset(args[0])

        distribution = args[-1]

        try:
            self._data_dict[conditions] += distribution
        except:
            self._data_dict[conditions] = deepcopy(distribution)

# class for storing conditional probability distributions
class FragmentProbabilities(FragmentDistribution):

    # initialize, specifying a basis for input-wire conditions
    def __init__(self, init_basis = SIC, exit_basis = IZXY):
        assert( init_basis in [ SIC, IZXY, ZZXY ] )
        assert( exit_basis in [ SIC, IZXY, ZZXY ] )
        super().__init__()
        self.init_basis = init_basis
        self.exit_basis = exit_basis

    # retrieve a conditional distribution
    def __getitem__(self, conditions):
        if type(conditions) is tuple:
            assert( len(conditions) == 2 )
            conditions = self._combine_conds(*conditions)
        else:
            conditions = frozenset(conditions)

        # if we have the requested conditional distribution, return it
        try: return self._data_dict[conditions]
        except: None

        # otherwise, use available data to return the requested conditional distribution
        for is_input, oper, wire in conditions:

            if is_input:
                basis_to_check = self.init_basis
            else:
                basis_to_check = self.exit_basis

            if basis_to_check == SIC:
                if oper in op_basis_SIC: continue
                _get_dist = self._get_dist_SIC

            elif basis_to_check == IZXY:
                if oper in op_basis_IZXY: continue
                _get_dist = self._get_dist_IZXY

            elif basis_to_check == ZZXY:
                if oper in op_basis_ZZXY: continue
                _get_dist = self._get_dist_ZZXY

            vacancy = conditions.difference({(is_input,oper,wire)})
            def _dist(oper):
                return self[vacancy.union({(is_input,oper,wire)})]
            return _get_dist(oper, _dist)

    # return conditional distribution with conditions in the SIC basis
    def _get_dist_SIC(self, operator, _dist):

        # explicitly recognize the identity operator I by a string
        if operator == "I":
            return _dist((0,0,0)) * 2 # I = 2 * ( maximally mixed state )

        # explicitly recognize ZXY basis elements by a string
        if operator in state_vecs_ZXY.keys():
            return _dist(state_vecs_ZXY[operator])

        # return a distribution conditonal on an operator inside the bloch ball
        assert( len(operator) == 3 )
        return 1/4 * sum( ( 1 + 3 * np.dot(operator, vec) ) * _dist(idx)
                          for idx, vec in state_vecs_SIC.items() )

    # return conditional distribution with conditions in the {I,Z}ZXY bases
    def _get_dist_ZXY(self, operator, _dist, basis_completion):
        assert( basis_completion in [ IZXY, ZZXY ] )

        # explicitly recognize standard ZXY and SIC operators by a string
        if basis_completion == "IZXY":
            if operator == "-Z": return _dist("I") - _dist("+Z")
        else: # basis_completion == "ZZXY"
            if operator == "I": return _dist("+Z") + _dist("-Z")

        if operator == "-X": return _dist("I") - _dist("+X")
        if operator == "-Y": return _dist("I") - _dist("+Y")
        if operator in state_vecs_SIC.keys():
            return _dist(state_vecs_SIC[operator])

        # return a distribution conditonal on an operator inside the bloch ball
        assert( len(operator) == 3 )
        directions_ZXY = [ "+Z", "+X", "+Y" ]
        dist_ZXY = sum( val * _dist(direction)
                        for val, direction in zip(operator, directions_ZXY) )
        return dist_ZXY + _dist("I") * ( 1 - sum(operator) ) / 2

    # return conditional distribution with conditions in the IZXY basis
    def _get_dist_IZXY(self, operator, _dist):
        return self._get_dist_ZXY(operator, _dist, IZXY)

    # return conditional distribution with conditions in the ZZXY basis
    def _get_dist_ZZXY(self, operator, _dist):
        return self._get_dist_ZXY(operator, _dist, ZZXY)

# class for storing conditional amplitude distributions
class FragmentAmplitudes(FragmentDistribution):

    # retrieve a conditional distribution
    def __getitem__(self, conditions):
        if type(conditions) is tuple:
            assert( len(conditions) == 2 )
            conditions = self._combine_conds(*conditions)
        else:
            conditions = frozenset(conditions)

        # if we have the requested conditional distribution, return it
        try: return self._data_dict[conditions]
        except: None

        # otherwise, use available data to return the requested conditional distribution
        for is_input, state, wire in conditions:
            vacancy = conditions.difference({(is_input,state,wire)})
            def _amp_dist(state):
                return self[vacancy.union({(is_input,state,wire)})]

            theta, phi = get_bloch_angles(state)
            return ( self[_amp_dist(0)] * np.cos(theta/2) +
                     self[_amp_dist(1)] * np.sin(theta/2) * np.exp(1j*phi) )

    # convert into a FragmentProbabilities object
    def to_probabilities(self, init_basis = SIC, exit_basis = ZZXY, dtype = tf.float64):
        assert( init_basis in [ SIC, ZZXY ] )
        assert( exit_basis in [ SIC, ZZXY ] )

        def _dist_terms_SIC(oper, conjugate):
            assert( oper in op_basis_IZXY )
            sign = 1 if not conjugate else -1
            theta, phi = get_bloch_angles(state_vecs_SIC[oper])
            return [ ( 0, np.cos(theta/2) ), ( 1, np.exp(sign*1j*phi) * np.sin(theta/2) ) ]

        def _dist_terms_ZZXY(oper, conjugate):
            assert( oper in op_basis_ZZXY )
            if oper == "-Z":
                return [ ( 1, 1 ) ]
            if oper == "+Z":
                return [ ( 0, 1 ) ]
            if oper == "+X":
                return [ ( 0, 1/np.sqrt(2) ), ( 1, 1/np.sqrt(2) ) ]
            if oper == "+Y":
                sign = 1 if not conjugate else -1
                return [ ( 0, 1/np.sqrt(2) ), ( 1, sign*1j/np.sqrt(2) ) ]
            else: assert( False ) # something went badly wrong if we made it here

        _dist_terms = { SIC : _dist_terms_SIC,
                        ZZXY : _dist_terms_ZZXY }

        init_op_basis = op_basis[init_basis]
        exit_op_basis = op_basis[exit_basis]
        _init_dist_terms = _dist_terms[init_basis]
        _exit_dist_terms = _dist_terms[exit_basis]

        # identify all init/exit_wires
        for conditions in self.all_conditions():
            init_wires = { wire for is_input, _, wire in conditions if is_input }
            exit_wires = { wire for is_input, _, wire in conditions if not is_input }
            break

        probs = FragmentProbabilities(init_basis, exit_basis)
        for init_states in set_product(init_op_basis, repeat = len(init_wires)):
            prob_init_conds = { ( True, state, wire )
                                for state, wire in zip(init_states, init_wires) }

            init_terms = [ _init_dist_terms(state, False) for state in init_states ]

            for exit_states in set_product(exit_op_basis, repeat = len(exit_wires)):
                prob_exit_conds = { ( False, state, wire )
                                    for state, wire in zip(exit_states, exit_wires) }

                exit_terms = [ _exit_dist_terms(state, True) for state in exit_states ]

                state_amps = 0

                for init_bits_facs in set_product(*init_terms):
                    try: init_bits, init_facs = zip(*init_bits_facs)
                    except: init_bits, init_facs = [], []

                    init_fac = np.prod(init_facs)
                    amp_init_conds = { ( True, bit, wire )
                                       for bit, wire in zip(init_bits, init_wires)}

                    for exit_bits_facs in set_product(*exit_terms):
                        try: exit_bits, exit_facs = zip(*exit_bits_facs)
                        except: exit_bits, exit_facs = [], []

                        exit_fac = np.prod(exit_facs)
                        amp_exit_conds = { ( False, bit, wire )
                                           for bit, wire in zip(exit_bits, exit_wires)}

                        fac = init_fac * exit_fac
                        state_amps += fac * self[amp_init_conds, amp_exit_conds]

                cond_probs = tf.cast(abs(state_amps)**2, dtype = dtype)
                probs.add(prob_init_conds, prob_exit_conds, cond_probs)

        return probs
