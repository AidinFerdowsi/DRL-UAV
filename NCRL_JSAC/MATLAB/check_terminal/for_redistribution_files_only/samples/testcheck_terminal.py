#!/usr/bin/env python
"""
Sample script that uses the check_terminal module created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

from __future__ import print_function
import check_terminal
import matlab

my_check_terminal = check_terminal.initialize()

TIn = matlab.double([180.0], size=(1, 1))
vmaxIn = matlab.double([20.0], size=(1, 1))
EIn = matlab.double([10002.0, 100100.0], size=(1, 2))
x0In = matlab.double([5.0], size=(1, 1))
y0In = matlab.double([100.0], size=(1, 1))
xfIn = matlab.double([587.0], size=(1, 1))
yfIn = matlab.double([879.0], size=(1, 1))
xiIn = matlab.double([100.0, 627.0], size=(1, 2))
yiIn = matlab.double([123.0, 454.0], size=(1, 2))
hIn = matlab.double([80.0], size=(1, 1))
lambdaIn = matlab.double([0.3, 0.7], size=(1, 2))
uIn = matlab.double([2.0, 1.0, 2.0, 2.0], size=(1, 4))
nIn = matlab.double([2.0], size=(1, 1))
terminalOut = my_check_terminal.check_terminal(TIn, vmaxIn, EIn, x0In, y0In, xfIn, yfIn, xiIn, yiIn, hIn, lambdaIn, uIn, nIn)
print(terminalOut, sep='\n')

my_check_terminal.terminate()
