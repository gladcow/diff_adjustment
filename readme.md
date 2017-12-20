Test script to test different difficulty adjustment algorithms in
different scenarios.

*Usage*

usage: Run a mining simulation [-h] [-a algo] [-s scenario] [-r seed]
                               [-n count]

optional arguments:
  -h, --help            show this help message and exit
  -a algo, --algo algo  algorithm choice
  -s scenario, --scenario scenario
                        scenario choice
  -r seed, --seed seed  random seed
  -n count, --count count
                        count of simuls to run

Available algorithms:
    'cdgw3-24':  24-blocks, current Dash
    'fdgw3-24':  24-blocks, fixed Dash
    'cdgw3-144':  1 full day,  current Dash
    'fdgw3-144':  1 full day,  fixed Dash
    'xmr': Monero
    'cdho': based on using critically damped harmonic oscillator
    'pid': based on using PID-controller
    'sa': "simple align" method (planned block time + last block time = 2 * target block time)

Available scenarios:
    'const': constant network hashrate
    'random_oscillations': random oscillations of the network hashrate
                           around given value
    'increase': increase network hashrate to given value (2 x initial)
    'decrease': decrease network hashrate to given value (0.5 x initial)
    'inout': some part of the whole network is in and out periodically
