import time
import sys
import numpy as np
import simcore as sc
import controllers as ct

profs = sc.make_profiles()
net = sc.Network()
for scen in sys.argv[1:] or ['S2']:
    qb = sc.scenario_boundary_flows(sc.SCENARIOS[scen])
    C, plans, Ys = sc.webster_plan(net, qb)
    print(scen, 'Webster C', C, 'Y per junction', np.round(Ys, 3), 'plans', plans)
    q = net.analytic_flows(qb)
    print('approach flows veh/h', np.round(q).astype(int))
    for mk in ['FT-W', 'ACT', 'MP-C', 'MP-A', 'LPS']:
        if mk == 'FT-W':
            c = ct.FixedTimeWebster()
        elif mk == 'ACT':
            c = ct.Actuated()
        elif mk == 'MP-C':
            c = ct.CyclicMaxPressure(profs['P1'])
        elif mk == 'MP-A':
            c = ct.AcyclicMaxPressure(profs['P1'])
        else:
            c = ct.LocalProportional(profs['P1'])
        t0 = time.time()
        sim = sc.Sim(scen, 1, c, perception=profs['P1'])
        r = sim.run()
        dt = time.time() - t0
        print(f"{mk:5s} delay {r['delay']:7.1f} tt {r['travel_time']:7.1f} thr {r['throughput']:7.0f} "
              f"compl {r['completion']:.3f} q {r['queue_pcu']:6.1f} jain {r['jain']:.3f} gur {r['gur']:.3f} "
              f"blk {r['blocked_share']:.3f} spill {r['spill_share']:.3f} cyc {r['mean_cycle']:.0f} nV {sim.nV} ({dt:.1f}s)")
