#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyutilib.th as unittest
import os
import six

from pyomo.contrib.pynumero import numpy_available, scipy_available
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

import scipy.sparse as sp
import numpy as np

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run NLP tests")

import pyomo.environ as pyo
from pyomo.opt.base import WriterFactory
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP, AmplNLP
import tempfile

from scipy.sparse import coo_matrix

def create_pyomo_model1():
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    m.S = pyo.Set(initialize=[i+1 for i in range(9)])

    xb = dict()
    xb[1] = (-1,1)
    xb[2] = (-np.inf,2)
    xb[3] = (-3,np.inf)
    xb[4] = (-np.inf, np.inf)
    xb[5] = (-5,5)
    xb[6] = (-np.inf,6)
    xb[7] = (-7,np.inf)
    xb[8] = (-np.inf,np.inf)
    xb[9] = (-9,9)
    m.x = pyo.Var(m.S, initialize=1.0, bounds=lambda m,i: xb[i])

    cb = dict()
    cb[1] = (-1,1)
    cb[2] = (2,2)
    cb[3] = (-3,np.inf)
    cb[4] = (-np.inf, 4)
    cb[5] = (-5,5)
    cb[6] = (-6,-6)
    cb[7] = (-7,np.inf)
    cb[8] = (-np.inf,8)
    cb[9] = (-9,9)

    def c_rule(m,i):
        return (cb[i][0], sum(i*j*m.x[j] for j in m.S), cb[i][1])
    m.c = pyo.Constraint(m.S, rule=c_rule)
    for i in m.S:
        m.dual.set_value(m.c[i], i)

    m.obj = pyo.Objective(expr=sum(i*j*m.x[i]*m.x[j] for i in m.S for j in m.S))
    return m

def create_pyomo_model2():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], domain=pyo.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.c1 = pyo.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.c2 = pyo.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.d1 = pyo.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.d2 = pyo.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.d3 = pyo.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = pyo.Objective(expr=m.x[2]**2)
    return m

@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestAslNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = create_pyomo_model1()
        temporary_dir = tempfile.mkdtemp()
        cls.filename = os.path.join(temporary_dir, "Pyomo_TestAslNLP")
        cls.pm.write(cls.filename+'.nl', io_options={"symbolic_solver_labels": True})

    @classmethod
    def tearDownClass(cls):
        # TODO: remove the nl files
        pass

    def test_create(self):
        anlp = AslNLP(self.filename)
        self.assertEquals(anlp.n_primals(),9)
        self.assertEquals(anlp._n_g, 9)
        self.assertEquals(anlp.n_eq_constraints(),2)
        self.assertEquals(anlp.n_ineq_constraints(),7)
        self.assertEquals(anlp.nnz_jacobian_eq(), 2*9)
        self.assertEquals(anlp.nnz_jacobian_ineq(), 7*9)
        self.assertEquals(anlp.nnz_hessian_lag(), 9*9)

        expected_primals_lb = np.asarray([-1, -np.inf, -3, -np.inf, -5, -np.inf, -7, -np.inf, -9], dtype=np.float64)
        expected_primals_ub = np.asarray([1, 2, np.inf, np.inf, 5, 6, np.inf, np.inf, 9], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_primals_lb, anlp.primals_lb()))
        self.assertTrue(np.array_equal(expected_primals_ub, anlp.primals_ub()))
        
        expected_condensed_primals_lb = [-1, -3, -5, -7, -9]
        expected_condensed_primals_ub = [1, 2, 5, 6, 9]
        self.assertTrue(np.array_equal(expected_condensed_primals_lb, anlp.primals_lb(condensed=True)))
        self.assertTrue(np.array_equal(expected_condensed_primals_ub, anlp.primals_ub(condensed=True)))

        expected_ineq_lb = np.asarray([-1, -3, -np.inf, -5, -7, -np.inf, -9], dtype=np.float64)
        expected_ineq_ub = np.asarray([1, np.inf, 4, 5, np.inf, 8, 9], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_ineq_lb, anlp.ineq_lb()))
        self.assertTrue(np.array_equal(expected_ineq_ub, anlp.ineq_ub()))

        expected_init_primals = np.ones(9)
        self.assertTrue(np.array_equal(expected_init_primals, anlp.init_primals()))
        expected_init_duals_ineq = np.asarray([1, 3, 4, 5, 7, 8, 9], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_init_duals_ineq, anlp.init_duals_ineq()))
        expected_init_duals_eq = np.asarray([2, 6], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_init_duals_eq, anlp.init_duals_eq()))

        t = anlp.create_new_vector('primals')
        self.assertTrue(t.size == 9)
        t = anlp.create_new_vector('primals_lb_condensed')
        self.assertTrue(t.size == 5)
        t = anlp.create_new_vector('primals_ub_condensed') 
        self.assertTrue(t.size == 5)
        t = anlp.create_new_vector('eq')
        self.assertTrue(t.size == 2)
        t = anlp.create_new_vector('ineq')
        self.assertTrue(t.size == 7)
        t = anlp.create_new_vector('duals_eq')
        self.assertTrue(t.size == 2)
        t = anlp.create_new_vector('duals_ineq')
        self.assertTrue(t.size == 7)
        t = anlp.create_new_vector('ineq_lb_condensed')
        self.assertTrue(t.size == 5)
        t = anlp.create_new_vector('ineq_ub_condensed')
        self.assertTrue(t.size == 5)

        expected_primals = [i+1 for i in range(9)]
        new_primals = np.asarray(expected_primals, dtype=np.float64)
        anlp.set_primals(new_primals)
        self.assertTrue(np.array_equal(expected_primals, anlp._primals))
        anlp.set_primals(np.ones(9))

        expected_duals_eq = [i+1 for i in range(2)]
        new_duals_eq = np.asarray(expected_duals_eq, dtype=np.float64)
        anlp.set_duals_eq(new_duals_eq)
        self.assertTrue(np.array_equal(expected_duals_eq, anlp._duals_eq))
        anlp.set_duals_eq(np.ones(2))

        expected_duals_ineq = [i+1 for i in range(7)]
        new_duals_ineq = np.asarray(expected_duals_ineq, dtype=np.float64)
        anlp.set_duals_ineq(new_duals_ineq)
        self.assertTrue(np.array_equal(expected_duals_ineq, anlp._duals_ineq))
        anlp.set_duals_ineq(np.ones(7))

        # objective function
        expected_objective = sum((i+1)*(j+1) for i in range(9) for j in range(9))
        self.assertEquals(expected_objective, anlp.evaluate_objective())
        # change the value of the primals
        anlp.set_primals(2.0*np.ones(9))
        expected_objective = sum(2.0**2*(i+1)*(j+1) for i in range(9) for j in range(9))
        self.assertEquals(expected_objective, anlp.evaluate_objective())
        anlp.set_primals(np.ones(9))

        # gradient of the objective
        expected_gradient = np.asarray([2*sum((i+1)*(j+1) for j in range(9)) for i in range(9)], dtype=np.float64)
        grad_obj = anlp.evaluate_grad_objective()
        self.assertTrue(np.array_equal(expected_gradient, grad_obj))
        # test inplace
        grad_obj = np.ones(9)
        anlp.evaluate_grad_objective(out=grad_obj)
        self.assertTrue(np.array_equal(expected_gradient, grad_obj))
        # change the value of the primals
        anlp.set_primals(2.0*np.ones(9))
        expected_gradient = np.asarray([2*2*sum((i+1)*(j+1) for j in range(9)) for i in range(9)], dtype=np.float64)
        grad_obj = np.ones(9)
        anlp.evaluate_grad_objective(out=grad_obj)
        self.assertTrue(np.array_equal(expected_gradient, grad_obj))
        anlp.set_primals(np.ones(9))

        # equality constraints
        con_eq = anlp.evaluate_eq_constraints()
        expected_con_eq = np.asarray([88, 276], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_con_eq, con_eq))
        # test inplace
        con_eq = np.zeros(2)
        anlp.evaluate_eq_constraints(out=con_eq)
        self.assertTrue(np.array_equal(expected_con_eq, con_eq))
        # change the value of the primals
        anlp.set_primals(2.0*np.ones(9))
        con_eq = np.zeros(2)
        anlp.evaluate_eq_constraints(out=con_eq)
        expected_con_eq = np.asarray([2*(88+2)-2, 2*(276-6)+6], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_con_eq, con_eq))
        anlp.set_primals(np.ones(9))

        # inequality constraints
        con_ineq = anlp.evaluate_ineq_constraints()
        expected_con_ineq = np.asarray([45, 3*45, 4*45, 5*45, 7*45, 8*45, 9*45], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_con_ineq, con_ineq))
        # test inplace
        con_ineq = np.zeros(7)
        anlp.evaluate_ineq_constraints(out=con_ineq)
        self.assertTrue(np.array_equal(expected_con_ineq, con_ineq))
        # change the value of the primals
        anlp.set_primals(2.0*np.ones(9))
        con_ineq = np.zeros(7)
        anlp.evaluate_ineq_constraints(out=con_ineq)
        expected_con_ineq = 2.0*expected_con_ineq
        self.assertTrue(np.array_equal(expected_con_ineq, con_ineq))
        anlp.set_primals(np.ones(9))

        # jacobian of equality constraints
        jac_eq = anlp.evaluate_jacobian_eq()
        dense_jac_eq = jac_eq.todense()
        expected_jac_eq = np.asarray([[2, 4, 6, 8, 10, 12, 14, 16, 18],
                                      [6, 12, 18, 24, 30, 36, 42, 48, 54]], dtype=np.float64)
        self.assertTrue(np.array_equal(dense_jac_eq, expected_jac_eq))
        # test inplace
        jac_eq.data = 0*jac_eq.data
        anlp.evaluate_jacobian_eq(out=jac_eq)
        dense_jac_eq = jac_eq.todense()
        self.assertTrue(np.array_equal(dense_jac_eq, expected_jac_eq))
        # change the value of the primals
        # ToDo: not a great test since this problem is linear
        anlp.set_primals(2.0*np.ones(9))
        anlp.evaluate_jacobian_eq(out=jac_eq)
        dense_jac_eq = jac_eq.todense()
        self.assertTrue(np.array_equal(dense_jac_eq, expected_jac_eq))

        # jacobian of inequality constraints
        jac_ineq = anlp.evaluate_jacobian_ineq()
        dense_jac_ineq = jac_ineq.todense()
        expected_jac_ineq = [ [(i)*(j) for j in range(1,10)] for i in [1, 3, 4, 5, 7, 8, 9] ]
        expected_jac_ineq = np.asarray(expected_jac_ineq, dtype=np.float64)
        self.assertTrue(np.array_equal(dense_jac_ineq, expected_jac_ineq))
        # test inplace
        jac_ineq.data = 0*jac_ineq.data
        anlp.evaluate_jacobian_ineq(out=jac_ineq)
        dense_jac_ineq = jac_ineq.todense()
        self.assertTrue(np.array_equal(dense_jac_ineq, expected_jac_ineq))
        # change the value of the primals
        # ToDo: not a great test since this problem is linear
        anlp.set_primals(2.0*np.ones(9))
        anlp.evaluate_jacobian_ineq(out=jac_ineq)
        dense_jac_ineq = jac_ineq.todense()
        self.assertTrue(np.array_equal(dense_jac_ineq, expected_jac_ineq))

        # hessian
        hess = anlp.evaluate_hessian_lag()
        dense_hess = hess.todense()
        expected_hess = [ [2.0*i*j for j in range(1, 10)] for i in range(1,10) ]
        expected_hess = np.asarray(expected_hess, dtype=np.float64)
        self.assertTrue(np.array_equal(dense_hess, expected_hess))
        # test inplace
        hess.data = np.zeros(len(hess.data))
        anlp.evaluate_hessian_lag(out=hess)
        dense_hess = hess.todense()
        self.assertTrue(np.array_equal(dense_hess, expected_hess))
        # change the value of the primals
        anlp.set_primals(2.0*np.ones(9))
        anlp.evaluate_hessian_lag(out=hess)
        dense_hess = hess.todense()
        self.assertTrue(np.array_equal(dense_hess, expected_hess))
        
@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestAmplNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # test problem
        cls.pm2 = create_pyomo_model2()
        temporary_dir = tempfile.mkdtemp()
        cls.filename = os.path.join(temporary_dir, "Pyomo_TestAmplNLP")
        cls.pm2.write(cls.filename+'.nl', io_options={"symbolic_solver_labels": True})
        cls.nlp = AmplNLP(cls.filename+'.nl',
                          row_filename=cls.filename+'.row',
                          col_filename=cls.filename+'.col')

    @classmethod
    def tearDownClass(cls):
        # TODO: remove the nl files
        pass

    def test_names(self):
        # Note: order may not be the same as "expected"
        expected_variable_names = ['x[1]', 'x[2]', 'x[3]']
        variable_names = self.nlp.variable_names()
        self.assertEqual(len(expected_variable_names),len(variable_names))
        for i in range(len(expected_variable_names)):
            self.assertTrue(expected_variable_names[i] in variable_names)

        # Note: order may not be the same as "expected"
        expected_eq_constraint_names = ['c1', 'c2']
        eq_constraint_names = self.nlp.eq_constraint_names()
        self.assertEqual(len(expected_eq_constraint_names),len(eq_constraint_names))
        for i in range(len(expected_eq_constraint_names)):
            self.assertTrue(expected_eq_constraint_names[i] in eq_constraint_names)

        # Note: order may not be the same as "expected"
        expected_ineq_constraint_names = ['d1', 'd2', 'd3']
        ineq_constraint_names = self.nlp.ineq_constraint_names()
        self.assertEqual(len(expected_ineq_constraint_names),len(ineq_constraint_names))
        for i in range(len(expected_ineq_constraint_names)):
            self.assertTrue(expected_ineq_constraint_names[i] in ineq_constraint_names)

    def test_idxs(self):
        # Note: order may not be the same as expected
        variable_idxs = list()
        variable_idxs.append(self.nlp.variable_idx('x[1]'))
        variable_idxs.append(self.nlp.variable_idx('x[2]'))
        variable_idxs.append(self.nlp.variable_idx('x[3]'))
        self.assertEquals(sum(variable_idxs), 3)

        # Note: order may not be the same as expected
        eq_constraint_idxs = list()
        eq_constraint_idxs.append(self.nlp.eq_constraint_idx('c1'))
        eq_constraint_idxs.append(self.nlp.eq_constraint_idx('c2'))
        self.assertEquals(sum(eq_constraint_idxs), 1)

        # Note: order may not be the same as expected
        ineq_constraint_idxs = list()
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('d1'))
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('d2'))
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('d3'))
        self.assertEquals(sum(ineq_constraint_idxs), 3)

if __name__ == '__main__':
    TestAslNLP.setUpClass()
    t = TestAslNLP()
    t.test_create()
    
