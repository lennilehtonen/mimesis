#!/usr/bin/env python3
"""Build app/index.html = shell + screens/*.html fragments (record, episode, multiply, ship).
   python3 rakenna.py"""
import os
K = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(os.path.dirname(K), 'web')
osat = []
for id in ['record', 'episode', 'multiply', 'ship']:
    p = os.path.join(A, 'screens', id + '.html')
    osat.append(open(p).read() if os.path.exists(p) else f'<section id="scr-{id}" class="screen"></section>')
shell = open(os.path.join(A, 'shell.html')).read()
open(os.path.join(A, 'index.html'), 'w').write(shell.replace('<!--SCREENS-->', '\n'.join(osat)))
print('web/index.html')
