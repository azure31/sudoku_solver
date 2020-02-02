# -*- coding: utf-8 -*-
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash
from werkzeug import secure_filename
from extract_board import extract_board
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, SelectField
from wtforms.validators import DataRequired, NumberRange 
from jinja2 import Template
from sudoku import print_board, solve_board, print_board_jinja
import numpy as np

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
#UPLOAD_FOLDER = '/home/ITTIAM/100715/Codes/keras-flask-deploy-webapp/sudoku/static/uploads'
UPLOAD_FOLDER = 'C:\\Users\\Prahita\\Desktop\\DS\\SE\\sudoku_solver\\sudoku\\static\\uploads'
 
# Declare a flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.debug = True
app.config['SECRET_KEY'] = "hahagotcha"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
board = np.zeros((9,9), np.uint8)


class SudokuForm(FlaskForm):

    row = SelectField('Row Index', choices = list(range(1,10,1)), validators = [DataRequired()])
    col = SelectField('Col Index', choices = list(range(1,10,1)), validators = [DataRequired()])
    val = SelectField('Value', choices = list(range(0,10,1)), validators = [DataRequired()])
    submit = SubmitField('Update')
    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('filename')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('board1.html', filename=filename)
        else:
            flash("Pick .jpg, .jpeg or .png image format")
    return render_template('upload.html')


@app.route('/detect_board', methods=['GET', 'POST'])
def detect_board():
    form = SudokuForm()
    global board
    if request.method == 'POST':
        if "Detect Board" in request.form:
            filename = request.form.get('filename')
            remove_lines = request.form.get('remove_lines')
            if(filename is not None):
                filepath= os.path.join(app.config['UPLOAD_FOLDER'], filename)
                board = extract_board(filepath, remove_lines=remove_lines)
                return render_template('board2.html', filename=filename, arr=board, form=SudokuForm())
            else:
                flash("No filename selected. Try again!")
    
    if request.method == 'GET':
        filename = "sample.png"
        return render_template('board1.html', filename=filename)

    return render_template('upload.html')
           

@app.route('/update_board', methods=['GET', 'POST'])
def update_board():
    form = SudokuForm()
    if (request.method == 'POST'):
        filename = request.form.get('filename')
        row = int(request.form.get('row'))
        col = int(request.form.get('col'))
        val = int(request.form.get('val'))
        board[row-1][col-1] = val    
        flash(f"Value at ({row},{col}) updated to {val}", category='success')    
    return render_template('board2.html', filename=filename, arr=board, form=SudokuForm())
           

@app.route('/display_board/<filename>')
def uploaded_file(filename):
    form = SudokuForm()
    return render_template('form2.html', filename=filename, form=form)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/solve', methods=['POST'])
def solve():
    if request.method == 'POST':
        if "Solve" in request.form:
            solved = solve_board(board)
            filename = request.form.get('filename')
            if(not solved is False):
                #global board
                #board = solved[1]
                flash("Sudoku Solved!")
                return render_template('board3.html', filename=filename, arr=solved[1])
            else:
                flash("Could not solve. Check if there are any errors in the board", 'failure')
                return "Could not solve. Check if there are any errors in the board"
    return render_template('board2.html', filename=filename, arr=board, form=SudokuForm())

  
