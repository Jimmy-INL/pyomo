#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
This module defines the classes that provide an NLP interface based on
the Ampl Solver Library (ASL) implementation
"""
import pyomo
import pyomo.environ as aml
from ampl_nlp import AslNLP

from scipy.sparse import coo_matrix, csr_matrix
import abc
import numpy as np
import tempfile
import os
import six
import shutil

__all__ = ['PyomoNLP']

class PyomoNLP(AslNLP):
    """
    Pyomo nonlinear program interface


    Attributes
    ----------
    _varToIndex: pyomo.core.kernel.ComponentMap
        Map from variable to variable idx
    _conToIndex: pyomo.core.kernel.ComponentMap
        Map from constraint to constraint idx

    Parameters
    ----------
    model: pyomo.environ.ConcreteModel
        Pyomo concrete model

    """

    def __init__(self, model):
        temp_dir = tempfile.mkdtemp()
        try:
            filename = os.path.join(temp_dir, "pynumero_pyomo")
            objectives = model.component_map(aml.Objective, active=True)
            if len(objectives) == 0:
                model._dummy_obj = aml.Objective(expr=0.0)

            model.write(filename+'.nl', 'nl', io_options={"symbolic_solver_labels": True})

            fname, symbolMap = pyomo.opt.WriterFactory('nl')(model, filename, lambda x:True, {})
            varToIndex = pyomo.core.kernel.component_map.ComponentMap()
            conToIndex = pyomo.core.kernel.component_map.ComponentMap()
            for name, obj in six.iteritems(symbolMap.bySymbol):
                if name[0] == 'v':
                    varToIndex[obj()] = int(name[1:])
                elif name[0] == 'c':
                    conToIndex[obj()] = int(name[1:])

            self._varToIndex = varToIndex
            self._conToIndex = conToIndex

            nl_file = filename+".nl"

            super(PyomoNLP, self).__init__(nl_file)

            # keep pyomo model in cache
            self._model = model

        finally:
            shutil.rmtree(temp_dir)

    @property
    def model(self):
        """
        Return optimization model
        """
        return self._model

    def grad_objective(self, x, out=None, **kwargs):
        """Returns gradient of the objective function evaluated at x

        Parameters
        ----------
        x: numpy.ndarray
            Array with values of primal variables.
        out: numpy.ndarray, optional
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Other Parameters
        ----------------
        subset_variables: list, optional
            List of pyomo variables to include in gradient (default None).
            Default includes all variables

        Returns
        -------
        numpy.ndarray

        """
        subset_variables = kwargs.pop('subset_variables', None)

        if subset_variables is None:
            return super(PyomoNLP, self).grad_objective(x,
                                                        out=out,
                                                        **kwargs)

        if out is not None:
            msg = 'out not supported with subset of variables'
            raise RuntimeError(msg)
        df = super(PyomoNLP, self).grad_objective(x, out=out, **kwargs)

        var_indices = []
        for v in subset_variables:
            if v.is_indexed():
                for vd in v.values():
                    var_id = self._varToIndex[vd]
                    var_indices.append(var_id)
            else:
                var_id = self._varToIndex[v]
                var_indices.append(var_id)
        return df[var_indices]

    def evaluate_g(self, x, out=None, **kwargs):
        """Return general inequality constraints evaluated at x

        Parameters
        ----------
        x: numpy.ndarray
            Array with values of primal variables.
        out: numpy.ndarray, optional
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Other Parameters
        ----------------
        subset_constraints: list, optional
            List of pyomo constraints to include in evaluated vector (default None).
            Default includes all constraints

        Returns
        -------
        numpy.ndarray

        """
        subset_constraints = kwargs.pop('subset_constraints', None)

        if subset_constraints is None:
            return super(PyomoNLP, self).evaluate_g(x,
                                                    out=out,
                                                    **kwargs)

        if out is not None:
            msg = 'out not supported with subset of constraints'
            raise RuntimeError(msg)

        res = super(PyomoNLP, self).evaluate_g(x,
                                               out=out,
                                               **kwargs)
        con_indices = []
        for c in subset_constraints:
            if c.is_indexed():
                for cd in c.values():
                    con_id = self._conToIndex[cd]
                    con_indices.append(con_id)
            else:
                con_id = self._conToIndex[c]
                con_indices.append(con_id)
        return res[con_indices]

    def jacobian_g(self, x, out=None, **kwargs):
        """Returns the Jacobian of the general inequalities evaluated at x

        Parameters
        ----------
        x: numpy.ndarray
            Array with values of primal variables.
        out: scipy.sparse.coo_matrix, optional
            Output matrix with the structure of the jacobian already defined.

        Other Parameters
        ----------------
        subset_variables: list, optional
            List of pyomo variables to include in evaluated jacobian (default None).
            Default includes all variables
        subset_constraints: list, optional
            List of pyomo constraints to include in evaluated jacobian (default None).
            Default includes all constraints

        Returns
        -------
        scipy.sparse.coo_matrix

        """
        subset_variables = kwargs.pop('subset_variables', None)
        subset_constraints = kwargs.pop('subset_constraints', None)

        if subset_variables is None and subset_constraints is None:
            return super(PyomoNLP, self).jacobian_g(x,
                                                    out=out,
                                                    **kwargs)

        if out is not None:
            msg = 'out not supported with subset of ' \
                  'variables or constraints'
            raise RuntimeError(msg)

        subset_vars = False
        if subset_variables is not None:
            var_indices = []
            for v in subset_variables:
                if v.is_indexed():
                    for vd in v.values():
                        var_id = self._varToIndex[vd]
                        var_indices.append(var_id)
                else:
                    var_id = self._varToIndex[v]
                    var_indices.append(var_id)
            indices_vars = var_indices
            subset_vars = True

        subset_constr = False
        if subset_constraints is not None:
            con_indices = []
            for c in subset_constraints:
                if c.is_indexed():
                    for cd in c.values():
                        con_id = self._conToIndex[cd]
                        con_indices.append(con_id)
                else:
                    con_id = self._conToIndex[c]
                    con_indices.append(con_id)
            indices_constraints = con_indices
            subset_constr = True

        if subset_vars:
            jcols_bool = np.isin(self._jcols_jac_g, indices_vars)
            ncols = len(indices_vars)
        else:
            jcols_bool = np.ones(self.nnz_jacobian_g, dtype=bool)
            ncols = self.nx

        if subset_constr:
            irows_bool = np.isin(self._irows_jac_g, indices_constraints)
            nrows = len(indices_constraints)
        else:
            irows_bool = np.ones(self.nnz_jacobian_g, dtype=bool)
            nrows = self.ng

        vals_bool = irows_bool * jcols_bool
        vals_indices = np.where(vals_bool)

        jac = super(PyomoNLP, self).jacobian_g(x, out=out, **kwargs)
        data = jac.data[vals_indices]

        # map indices to new indices
        new_col_indices = self._jcols_jac_g[vals_indices]
        if subset_vars:
            old_col_indices = self._jcols_jac_g[vals_indices]
            vid_to_nvid = {vid: idx for idx, vid in enumerate(indices_vars)}
            new_col_indices = np.array([vid_to_nvid[vid] for vid in old_col_indices])

        new_row_indices = self._irows_jac_g[vals_indices]
        if subset_constraints:
            old_const_indices = self._irows_jac_g[vals_indices]
            cid_to_ncid = {cid: idx for idx, cid in enumerate(indices_constraints)}
            new_row_indices = np.array([cid_to_ncid[cid] for cid in old_const_indices])

        return coo_matrix((data, (new_row_indices, new_col_indices)),
                         shape=(nrows, ncols))

    def hessian_lag(self, x, y, out=None, **kwargs):
        """Return the Hessian of the Lagrangian function evaluated at x and y

        Parameters
        ----------
        x: numpy.ndarray
            Array with values of primal variables.
        y: numpy.ndarray
            Array with values of dual variables.
        out: scipy.sparse.coo_matrix, optional
            Output matrix with the structure of the hessian already defined.

        Other Parameters
        ----------------
        eval_f_c: bool, optional
            True if objective and contraints need to be reevaluated (default True).
        obj_factor: float64, optional
            Factor used to scale objective function (default 1.0)
        subset_variables_row: list, optional
            List of pyomo variables to include in rows of evaluated hessian (default None).
            Default includes all variables
        subset_variables: list, optional
            List of pyomo variables to include in columns of evaluated hessian (default None).
            Default includes all variables

        Returns
        -------
        scipy.sparse.coo_matrix

        """
        subset_variables_row = kwargs.pop('subset_variables_row', None)
        subset_variables_col = kwargs.pop('subset_variables_col', None)

        if subset_variables_row is None and subset_variables_col is None:
            return super(PyomoNLP, self).hessian_lag(x,
                                                     y,
                                                     out=out,
                                                     **kwargs)

        if out is not None:
            msg = 'out not supported with subset of variables'
            raise RuntimeError(msg)

        subset_cols = False
        if subset_variables_col is not None:
            var_indices_cols = []
            for v in subset_variables_col:
                if v.is_indexed():
                    for vd in v.values():
                        var_id = self._varToIndex[vd]
                        var_indices_cols.append(var_id)
                else:
                    var_id = self._varToIndex[v]
                    var_indices_cols.append(var_id)

            indices_cols = var_indices_cols
            subset_cols = True

        subset_rows = False
        if subset_variables_row is not None:
            var_indices_rows = []
            for v in subset_variables_row:
                if v.is_indexed():
                    for vd in v.values():
                        var_id = self._varToIndex[vd]
                        var_indices_rows.append(var_id)
                else:
                    var_id = self._varToIndex[v]
                    var_indices_rows.append(var_id)
            indices_rows = var_indices_rows
            subset_rows = True

        if subset_cols:
            jcols_bool = np.isin(self._jcols_hess, indices_cols)
            ncols = len(indices_cols)
        else:
            jcols_bool = np.ones(self.nnz_hessian_lag, dtype=bool)
            ncols = self.nx

        if subset_rows:
            irows_bool = np.isin(self._irows_hess, indices_rows)
            nrows = len(indices_rows)
        else:
            irows_bool = np.ones(self.nnz_hessian_lag, dtype=bool)
            nrows = self.nx

        vals_bool = irows_bool * jcols_bool
        vals_indices = np.where(vals_bool)

        hess = super(PyomoNLP, self).hessian_lag(x, y, out=out, **kwargs)
        data = hess.data[vals_indices]

        # map indices to new indices
        new_col_indices = self._jcols_hess[vals_indices]
        if subset_cols:
            old_col_indices = self._jcols_hess[vals_indices]
            vid_to_nvid = {vid: idx for idx, vid in enumerate(indices_cols)}
            new_col_indices = np.array([vid_to_nvid[vid] for vid in old_col_indices])

        new_row_indices = self._irows_hess[vals_indices]
        if subset_rows:
            old_row_indices = self._irows_hess[vals_indices]
            cid_to_ncid = {cid: idx for idx, cid in enumerate(indices_rows)}
            new_row_indices = np.array([cid_to_ncid[cid] for cid in old_row_indices])

        return coo_matrix((data, (new_row_indices, new_col_indices)),
                         shape=(nrows, ncols))

    def variable_order(self):
        """Returns ordered list with names of primal variables"""
        #return {idx: v.name for v,idx in self._varToIndex.items()}
        var_order = [None] * self.nx
        for v, idx in self._varToIndex.items():
            var_order[idx] = v.name
        return var_order

    def variables(self):
        """Returns a dictionary of the primal variables indexed by their order"""
        return {idx:v for v,idx in self._varToIndex.items()}
    
    def constraint_order(self):
        """Returns ordered list with names of constraints"""
        con_order = [None] * self.ng
        for c, idx in self._conToIndex.items():
            con_order[idx] = c.name
        return con_order

    def constraints(self):
        """Returns a dictionary of the constraints indexed by their order"""
        return {idx:c for c,idx in self._conToIndex.items()}
        
    def variable_idx(self, var):
        """
        Returns index of variable in nlp.x

        Parameters
        ----------
        var: pyomo.Var
            Pyomo variable

        Returns
        -------
        int

        """
        if var.is_indexed():
            raise RuntimeError("Var must of type VarData (not indexed)")
        return self._varToIndex[var]

    def constraint_idx(self, constraint):
        """
        Returns index of constraint in nlp.g

        Parameters
        ----------
        con_name: pyomo.Constraint
            Pyomo Constraint

        Returns
        -------
        int

        """
        if constraint.is_indexed():
            raise RuntimeError("Constraint must be of type ConstraintData (not indexed)")
        return self._conToIndex[constraint]

