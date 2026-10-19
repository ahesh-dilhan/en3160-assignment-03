"""Export the executed notebook through LaTeX, preserving its recorded outputs."""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
import nbformat
import pypandoc

PREAMBLE=r'''\documentclass[10pt,a4paper]{article}
\usepackage[top=15mm,bottom=17mm,left=18mm,right=18mm,headheight=13pt,headsep=5mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,calc,xcolor}
\usepackage{fancyhdr,titlesec,caption,listings,microtype,float}
\usepackage[hidelinks]{hyperref}
\hypersetup{colorlinks=true,allcolors=black,pdftitle={EN3160 Assignment 3: Neural Networks},pdfauthor={Dilhan W.A.}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small EN3160 -- Assignment 03}
\fancyhead[R]{\small 230145E}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.13}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\titleformat{\section}{\large\bfseries}{}{0em}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{}{0em}{}
\titlespacing*{\section}{0pt}{9pt}{5pt}
\titlespacing*{\subsection}{0pt}{7pt}{4pt}
\captionsetup{font=footnotesize,labelfont=bf}
\lstset{basicstyle=\ttfamily\scriptsize,keywordstyle=\bfseries,commentstyle=\itshape,
 stringstyle=\ttfamily,frame=tb,rulecolor=\color{black},showstringspaces=false,
 breaklines=true,columns=fullflexible,keepspaces=true,aboveskip=5pt,belowskip=5pt,
 framesep=3pt,language=Python}
\color{black}
\begin{document}
\thispagestyle{plain}
\begin{minipage}[c]{.13\linewidth}
\includegraphics[height=22mm]{../assets/uom_logo.png}
\end{minipage}\hfill
\begin{minipage}[c]{.84\linewidth}
{\small\bfseries University of Moratuwa, Sri Lanka}\par
{\footnotesize Department of Electronic \& Telecommunication Engineering}\par
\vspace{3pt}
{\Large\bfseries EN3160 -- Neural Networks}\par
{\small Assignment 03\quad$\vert$\quad Dilhan W.A.\quad$\vert$\quad 230145E}\par
{\footnotesize October 19, 2026}
\end{minipage}
\par\vspace{5pt}\hrule\vspace{4pt}
{\footnotesize\textbf{Repository:} \url{https://github.com/ahesh-dilhan/en3160-assignment-03}}\par
'''

def ascii_text(text):
    for a,b in {'→':' -> ','−':'-','–':'--','—':'---','×':' x ','·':' | ','’':"'",'“':'"','”':'"','…':'...'}.items():
        text=text.replace(a,b)
    return text

def markdown(text):
    text=ascii_text(text)
    pagebreak='<!-- pagebreak -->' in text
    text=text.replace('<!-- pagebreak -->','')
    result=pypandoc.convert_text(text,'latex',format='markdown+pipe_tables+tex_math_dollars',
                                 extra_args=['--wrap=none','--listings'])
    return ('\\clearpage\n' if pagebreak else '')+result

def listing(text,language=True):
    return '\\begin{lstlisting}'+('' if language else '[language={}]')+'\n'+ascii_text(text).rstrip()+'\n\\end{lstlisting}\n'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tectonic',default=os.environ.get('TECTONIC','tectonic'))
    args=parser.parse_args()
    notebook=nbformat.read(ROOT/'notebooks'/'230145E_a03.ipynb',as_version=4)
    pieces=[PREAMBLE]
    image_dir=ROOT/'report'/'notebook_images'
    image_dir.mkdir(exist_ok=True)
    image_count=0
    for index,cell in enumerate(notebook.cells):
        if cell.cell_type=='markdown':
            source=cell.source
            if index==0: source='This notebook'+source.split('This notebook',1)[1]
            pieces.append(markdown(source))
        elif cell.cell_type=='code':
            if 'hide-input' not in cell.metadata.get('tags',[]):pieces.append(listing(cell.source))
            for output in cell.get('outputs',[]):
                if output.output_type=='error':raise RuntimeError('Notebook contains a failed cell')
                if output.output_type=='stream':pieces.append(listing(output.text,False))
                else:
                    data=output.get('data',{})
                    if 'image/png' in data:
                        image_count+=1
                        name=f'figure_{image_count:02d}.png'
                        (image_dir/name).write_bytes(base64.b64decode(data['image/png']))
                        pieces.append(r'\begin{center}\includegraphics[width=\linewidth,height=.56\textheight,keepaspectratio]{notebook_images/'+name+r'}\end{center}'+'\n')
                    elif 'text/markdown' in data:pieces.append(markdown(data['text/markdown']))
                    elif 'text/plain' in data:pieces.append(listing(data['text/plain'],False))
    pieces.append('\\end{document}\n')
    tex=ROOT/'report'/'230145E_a03.tex'
    tex.write_text('\n'.join(pieces),encoding='utf-8')
    subprocess.run([args.tectonic,'--keep-logs','--outdir',str(ROOT/'output'),str(tex)],cwd=tex.parent,check=True)
    final=ROOT/'output'/'230145E_a03.pdf'
    shutil.copyfile(final,ROOT/'output'/'230145E_a02.pdf')
    with zipfile.ZipFile(ROOT/'output'/'230145E_a03_latex.zip','w',zipfile.ZIP_DEFLATED) as z:
        for directory in ['report','assets']:
            for path in sorted((ROOT/directory).rglob('*')):
                if path.is_file():z.write(path,path.relative_to(ROOT))
    print('Notebook figures:',image_count)
    print('Created',final)

if __name__=='__main__':main()
