#!/usr/bin/env python3

import numpy as np
import qiskit as qs

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.enable_v2_behavior()

from itertools import product as set_product
from copy import deepcopy

# define common strings to protect against typos
SIC = "SIC"
IZXY = "IZXY"
ZZXY = "ZZXY"
pauli = "pauli"

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

# define qubit operator vectors
pauli_vecs = { "I" : (1,0,0,0),
               "Z" : (0,1,0,0),
               "X" : (0,0,1,0),
               "Y" : (0,0,0,1) }

basis_ops_SIC = list(state_vecs_SIC.keys())
basis_ops_IZXY = [ "I", "+Z", "+X", "+Y" ]
basis_ops_ZZXY = [ "-Z", "+Z", "+X", "+Y" ]
basis_ops_pauli = list(pauli_vecs.keys())
basis_ops = { SIC : basis_ops_SIC,
              IZXY : basis_ops_IZXY,
              ZZXY : basis_ops_ZZXY,
              pauli : basis_ops_pauli }

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
        return ( cond for cond, _ in self )

    # iterate over all distributions
    def all_distributions(self):
        return ( dist for _, dist in self )

    # identify wires with init/exit conditions
    def all_wires(self):
        for conditions, _ in self: break
        return { cond[:2] for cond in conditions }
    def init_wires(self):
        return { wire for is_init, wire in self.all_wires() if is_init }
    def exit_wires(self):
        return { wire for is_init, wire in self.all_wires() if not is_init }

    # combine separate init/exit conditions into one set of conditions
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
        else: # len(args) == 2:
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
        for is_input, wire, oper in conditions:

            # get the basis in which we have stored data on the conditional distribution
            if is_input:
                dist_basis = self.init_basis
            else:
                dist_basis = self.exit_basis

            # if the requested operator at this stitch is already in the correct basis,
            # then we have already stored data in the requested basis and we don't need to
            # sum over different conditions, so move on to the next condition
            if oper in basis_ops[dist_basis]: continue

            # build a method to replace this condition with a new one
            vacancy = conditions.difference({(is_input,wire,oper)})
            def _dist(oper_str):
                return self[vacancy.union({(is_input,wire,oper_str)})]

            # explicitly recognize standard operators by a string
            if type(oper) is str:

                # convert operator string to a state / operator vector
                if oper in state_vecs.keys():
                    oper = state_vecs[oper]
                if oper in pauli_vecs.keys():
                    oper = pauli_vecs[oper]

            # convert 'oper' into an operator vector in the pauli basis
            if type(oper) is not str:

                # assert that we were given either a state vector or operator vector
                assert( len(oper) in [ 3, 4 ])

                # if we were given a state vector, convert it into an operator vector
                if len(oper) == 3:
                    oper = (1/2,) + tuple([ val/2 for val in oper ])

            # identify the method to use to get coefficients
            #   for the distributions that we collected data on
            _get_coeffs = { SIC : self._get_coeffs_SIC,
                            IZXY : self._get_coeffs_IZXY,
                            ZZXY : self._get_coeffs_ZZXY,
                            pauli : self._get_coeffs_pauli }

            # sum over distributions we collected data on with appropriate coefficients
            op_coeffs = _get_coeffs[dist_basis](oper)
            op_strs = basis_ops[dist_basis]
            return sum( coeff * _dist(op_str)
                        for coeff, op_str in zip(op_coeffs, op_strs)
                        if coeff != 0 )

    def _get_coeffs_SIC(self, op_vec):
        return [ 1/2 * ( op_vec[0] + 3 * np.dot(op_vec[1:], vec) )
                 for vec in state_vecs_SIC.values() ] # loop over all SIC state vectors

    def _get_coeffs_IZXY(self, op_vec):
        return [ op_vec[0] - sum(op_vec[1:]),  #  I
                 2 * op_vec[1],                # +Z
                 2 * op_vec[2],                # +X
                 2 * op_vec[3] ]               # +Y

    def _get_coeffs_ZZXY(self, op_vec):
        return [ op_vec[0] - op_vec[1] - op_vec[2] - op_vec[3], # -Z
                 op_vec[0] + op_vec[1] - op_vec[2] - op_vec[3], # +Z
                 2 * op_vec[2],                                 # +X
                 2 * op_vec[3] ]                                # +Y

    def _get_coeffs_pauli(self, op_vec): return op_vec

    # change the bases in which we store conditional probabilities
    def shuffle_bases(self, init_basis = pauli, exit_basis = pauli):
        new_probs = FragmentProbabilities()

        # identify wires with init/exit conditions
        init_wires = self.init_wires()
        exit_wires = self.exit_wires()

        # loop over all assignments of init/exit conditions in the appropriate bases
        for init_ops in set_product(basis_ops[init_basis], repeat = len(init_wires)):
            new_init_conds = [ ( True, init_wire, init_op )
                               for init_wire, init_op in zip(init_wires, init_ops) ]

            for exit_ops in set_product(basis_ops[exit_basis], repeat = len(exit_wires)):
                new_exit_conds = [ ( False, exit_wire, exit_op )
                                   for exit_wire, exit_op in zip(exit_wires, exit_ops) ]

                new_probs.add(new_init_conds, new_exit_conds,
                              self[new_init_conds, new_exit_conds])

        return new_probs

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

    # todo: write method to change bases for conditions

    # convert into a FragmentProbabilities object
    def to_probabilities(self, init_basis = pauli, exit_basis = pauli, dtype = tf.float64):
        assert( init_basis in basis_ops.keys() )
        assert( exit_basis in basis_ops.keys() )

        # save the bases in which we want to store init/exit conditions
        final_init_basis = init_basis
        final_exit_basis = exit_basis

        # we can only actually *compute* distributions in certain bases,
        # so if we weren't asked for one of those, choose one of them to use for now
        if init_basis not in [ SIC, ZZXY ]:
            init_basis = ZZXY
        if exit_basis not in [ SIC, ZZXY ]:
            exit_basis = ZZXY

        # identify computational basis states and coefficients for each SIC / ZZXY state
        def _dist_terms_SIC(oper, conjugate):
            assert( oper in basis_ops_SIC )
            sign = 1 if not conjugate else -1
            theta, phi = get_bloch_angles(state_vecs_SIC[oper])
            # | theta, phi > = cos(theta/2) |0> + exp(i phi) sin(theta/2) | 1 >
            return [ ( 0, np.cos(theta/2) ), ( 1, np.exp(sign*1j*phi) * np.sin(theta/2) ) ]

        def _dist_terms_ZZXY(oper, conjugate):
            assert( oper in basis_ops_ZZXY )
            if oper == "-Z": # | -Z > = 1 | 1 >
                return [ ( 1, 1 ) ]
            if oper == "+Z": # | +Z > = 1 | 0 >
                return [ ( 0, 1 ) ]
            if oper == "+X": # | +X > = ( | 0 > + | 1 > ) / sqrt(2)
                return [ ( 0, 1/np.sqrt(2) ), ( 1, 1/np.sqrt(2) ) ]
            if oper == "+Y": # | +Y > = ( | 0 > + i | 1 > ) / sqrt(2)
                sign = 1 if not conjugate else -1
                return [ ( 0, 1/np.sqrt(2) ), ( 1, sign*1j/np.sqrt(2) ) ]

        _dist_terms = { SIC : _dist_terms_SIC,
                        ZZXY : _dist_terms_ZZXY }

        # determine which basis of operators to use for init/exit conditions,
        # as well as corresponding computational basis states / coefficients
        init_basis_ops = basis_ops[init_basis]
        exit_basis_ops = basis_ops[exit_basis]
        _init_dist_terms = _dist_terms[init_basis]
        _exit_dist_terms = _dist_terms[exit_basis]

        # identify wires with init/exit conditions
        init_wires = self.init_wires()
        exit_wires = self.exit_wires()

        # initialize a conditional probability distribution
        probs = FragmentProbabilities(init_basis, exit_basis)

        # loop over all init/exit conditions (init_states/exit_states)
        for init_states in set_product(init_basis_ops, repeat = len(init_wires)):
            # conditions (for probs) corresponding to this choice of init_states
            prob_init_conds = { ( True, wire, state )
                                for wire, state in zip(init_wires, init_states) }

            # computational basis terms that contribute to this choice of init_states
            init_terms = [ _init_dist_terms(state, False) for state in init_states ]

            for exit_states in set_product(exit_basis_ops, repeat = len(exit_wires)):
                prob_exit_conds = { ( False, wire, state )
                                    for wire, state in zip(exit_wires, exit_states) }

                exit_terms = [ _exit_dist_terms(state, True) for state in exit_states ]

                state_amps = 0 # empty vector of amplitudes for these init/exit_states

                # looping over all contributing terms to this choice of init/exit_states
                for init_bits_facs in set_product(*init_terms):
                    try: init_bits, init_facs = zip(*init_bits_facs)
                    except: init_bits, init_facs = [], []

                    # scalar factor associated with this set of terms
                    init_fac = np.prod(init_facs)

                    # conditions (i.e. on the amplitude distribution) for to this term
                    amp_init_conds = { ( True, wire, bit )
                                       for wire, bit in zip(init_wires, init_bits)}

                    for exit_bits_facs in set_product(*exit_terms):
                        try: exit_bits, exit_facs = zip(*exit_bits_facs)
                        except: exit_bits, exit_facs = [], []

                        exit_fac = np.prod(exit_facs)
                        amp_exit_conds = { ( False, wire, bit )
                                           for wire, bit in zip(exit_wires, exit_bits)}

                        # add to the amplitudes for this choice of init/exit_states
                        fac = init_fac * exit_fac
                        state_amps += fac * self[amp_init_conds, amp_exit_conds]

                # having collected amplitudes, convert them to probabilities
                cond_probs = tf.cast(abs(state_amps)**2, dtype = dtype)
                probs.add(prob_init_conds, prob_exit_conds, cond_probs)

        # if we computed distributions in the same init/exit bases as we were asked for,
        #   then just return the conditional distribution we computed
        # otherwise, change bases appropriately
        if init_basis == final_init_basis and exit_basis == final_exit_basis:
            return probs
        else:
            return probs.shuffle_bases(final_init_basis, final_exit_basis)
