Preprocessing 101
=================
Source code for [carbonne.xyz/preprocessing101.html](http://carbonne.xyz/preprocessing101.html)

## Requirements
[Install Pandoc](http://pandoc.org/installing.html)

## Usage
* Create a reveal.js HTML page

    ```
    pandoc -s -S -t revealjs preprocessing101.md -o preprocessing101.html
    ```
* Create a PDF file (not optimised)

    ```
    pandoc preprocessing101.md --latex-engine=xelatex -o preprocessing101.pdf
    ```
    
Other possible usage [here](http://pandoc.org/demos.html)
