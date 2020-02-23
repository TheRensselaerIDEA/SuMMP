"""
The goal of this script will be to create an object-oriented approach to generate pdfs.
The object will collect and organize information in a manner that translates to elegant pdfs.
Main library incorperated will be reportlab. See: reportlab.com/opensource/
Documentation: https://www.reportlab.com/docs/reportlab-userguide.pdf

@author: Jose Figueroa
@email: josefigueroa168@gmail.com
"""

import pandas as pd
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

#TODO: Build custom ParagraphStyle class (Page 71 in doc) for customization of various items
#TODO: Index on __init__ script.

class PDF():
	def __init__(self, DIR="./"):
		"""
		Insert Arguments here
		"""
		self.doc = None
		self.title = None
		self.styles = getSampleStyleSheet()
		self.story = []
		self.DIR = DIR

	def newPDF(self, filename, **kwargs):
		"""
		Generate a new pdf document.

		Params
		------
		filename: str
			The name of your new pdf to generate
		For full list of arguments, see help(reportlab.platypus.SimpleDocTemplate)
		"""
		# TODO: Insert check for previously existing pdf
		# TODO: Error handle
		# TODO: Add directory to filename
		self.doc = SimpleDocTemplate(filename, **kwargs)


	def title(self, title, style=self.styles["title"]):
		"""
		Insert a title to pdf

		Params
		------
		title: str
			PDF title

		"""
		self.title = Paragraph(title, style)

	def table(self, dataframe, style=self.styles["Normal"]):
		"""
		Generates a table with given styling and adds it to the storyboard

		Params
		------
		dataframe: pd.Dataframe
			A dataframe, usually of a collection of results
		"""
		#TODO: Customizable alignment

		header = []
		row_list = []
		for heading in dataframe.columns:
			header.append(Paragraph("<para align=center>%s</para>" % heading, style))
		row_list.append(header)
		data = dataframe.to_dict("split")["data"]
		for record in data:
			row = [Paragraph("<para align=center>%s</para>" % item, style) for item in data]
			row_list.append(row)
		t = Table(row_list, hAlign="CENTER")
		t.setStyle(TableStyle([('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
					('BOX', (0,0), (-1,-1), 0.25, colors.black)]))
		self.story.append(t)

	def plot(self):
		# May be smart to generate separately and add pdf images (maybe def addImage?)
		print("stub")

	def build(self):
		# TODO: Append title to beginning
		print("stub")