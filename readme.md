  Test script to test different difficulty adjustment algorithms in
different scenarios.

## Usage

test_mining.py [-h] [-a algo] [-s scenario] [-r seed] [-n count]

 Optional arguments:
  -h, --help            show this help message and exit
  -a algo, --algo       algorithm choice
  -s scenario, --scenario  scenario choice
  -r seed, --seed       random seed
  -n count, --count     count of simuls to run

Available algorithms:
   * 'cdgw3-24':  24-blocks, current Dash
   * 'fdgw3-24':  24-blocks, fixed Dash
   * 'cdgw3-144':  1 full day,  current Dash
   * 'fdgw3-144':  1 full day,  fixed Dash
   * 'xmr': Monero
   * 'cdho': based on using critically damped harmonic oscillator
   * 'pid': based on using PID-controller
   * 'sa': "simple align" method (planned block time + last block time = 2 * target block time)
   * 'pe': 'proportional error' method (error in block time is reduced proportionaly, P-part of PID-controller)

Available scenarios:
   * 'const': constant network hashrate
   * 'random_oscillations': random oscillations of the network hashrate around given value
   * 'increase': increase network hashrate to given value (2 x initial)
   * 'decrease': decrease network hashrate to given value (0.5 x initial)
   * 'inout': some part of the whole network is in and out periodically

Conclusion:
    There is no valuable difference between this methods in different scenarios.
    PID-based control theory is not effective here because of the big error in
    "position detection" (we can "calculate" real current hashrate only from
    block times which is random with distribution that has a max at correct value
    but with big enough standard deviation). It is possible that stochastic control
    theory can be used, but I've failed to find good solution.

    About the errors in original dgw implementation (incorrect average calculation
    and one period lost in proportion): it doesn't allow timewrap attack (all values
    participates in difficulty calculation, result difficulty is smaller than it
    should be, but this error is compensated with feedback after the future time block
    timestamp become smaller,  so it is cause small shift from target block time,
    but this shift is much smaller than standard deviation of the random block time
    (it is approximately 1/24 part of the standard deviation), so there is no practical
    difference between current and fixed variant (we can see this in tests too).
    Fixing current algo requires hardfork, so it is better to leave it in current
    implementation.