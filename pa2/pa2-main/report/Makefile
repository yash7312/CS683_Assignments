all: main.pdf
	cp main.pdf ../Summary.pdf

main.pdf: main.tex preamble.tex $(wildcard sections/*.tex)
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -c

.PHONY: all clean
