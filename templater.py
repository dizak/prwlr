# -*- coding: utf-8 -*-


import jinja2 as jj2
import time


class HTML_Generator:
    """Saves results e.g. tables, figures, charts as an elegant html, using
    jinja2.

    Attribs:
        template_file (html with jinja2 vars): html template for embedding
        elements like charts or tables
        definitions_file (text file): stores lines showed in the final html
        under <Definitions> h1 tag. Each denifition is separated by newline.
        template (jinja2.enviroment.Template): loaded from
        HTML_Generator.template_file, containing vars vals passed into it
        template_rendered (unicode str): template rendered into regular unicode,
        save-ready
    """
    def __init__(self,
                 template_file,
                 definitions_file):
        self.template_file = template_file
        self.definitions_file = definitions_file
        self.template = None
        self.template_rendered = None

    def load_definitions(self):
        """Return HTML_Generator.definitions from
        HTML_Generator.definitions_file.
        """
        with open(self.definitions_file, "r") as fin:
            self.definitions = fin.readlines()
        self.definitions = [i.rstrip() for i in self.definitions]

    def load_template(self):
        """Load jinja2.environment.Template from HTML_Generator.template_file.
        Search path relative.
        """
        template_Loader = jj2.FileSystemLoader(searchpath="templates/")
        template_Env = jj2.Environment(loader=template_Loader)
        self.template = template_Env.get_template(self.template_file)

    def render_template(self,
                        name=None,
                        filters_pos=None,
                        filters_neg=None,
                        filters_neu=None,
                        num_prop_res=None,
                        num_prop_perm=None,
                        histogram_bins=None,
                        e_value=None,
                        histogram_gis_pos=None,
                        bivar_pos=None,
                        lin_regr_pos=None,
                        histogram_gis_neg=None,
                        bivar_neg=None,
                        lin_regr_neg=None,
                        histogram_gis_neu=None,
                        bivar_neu=None,
                        lin_regr_neu=None,
                        dataframe_pos=None,
                        dataframe_neg=None,
                        dataframe_neu=None,
                        results_type="chart",
                        skip_perm_res=False):
        """Retrun HTML_Generator.rendered_template with vals from passed vars.
        Args:
            name (str): name for results, displayed in most upper h2 tag
            filters_pos (list): passed from Ortho_Stats.filters_used. Displayed
            in ul tag in Filters article, under h2 DMF positive tag.
            filters_neg (list): passed from Ortho_Stats.filters_used. Displayed
            in ul tag in Filters article, under h2 DMF negative tag.
            filters_neu (list): passed from Ortho_Stats.filters_used. Displayed
            in ul tag in Filters article, under h2 DMF positive tag.
            num_prop_res (pandas.Series): passed from Ortho_Stats.num_prop_res.
            Values of num_prop_res are displayed in cells of Summary if
            results_type == <tbl>
            num_prop_perm (pandas.Series): passed from
            Ortho_Stats.num_prop_perm. Values of num_prop_perm are displayed in
            cells of Summary if results_type == <tbl>
            histogram_bins (pandas.DataFrame): sorted histogram bins from
            Ortho_Stats.num_prop_res. Used drawChart (JavaScript) func in
            template_file IMPORTANT: created externally. Will be included in
            one the classes of this script.
            e_value (int): passed from Ortho_Stats.e_value. Number of
            Ortho_Stats.inter_df permutations.
            histogram_gis_pos (str): path to png file, displayed in Plots
            section, above DMF positive h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            bivar_pos (str): path to png file, displayed in Plots
            section, above DMF positive h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            lin_regr_pos (str): path to png file, displayed in Plots
            section, above DMF positive h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            histogram_gis_neg (str): path to png file, displayed in Plots
            section, above DMF negative h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            bivar_neg (str): path to png file, displayed in Plots
            section, above DMF negative h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            lin_regr_neg (str): path to png file, displayed in Plots
            section, above DMF negative h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            histogram_gis_neu (str): path to png file, displayed in Plots
            section, above DMF neutral h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            bivar_neu (str): path to png file, displayed in Plots
            section, above neutral h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            bivar_neu (str): path to png file, displayed in Plots
            section, above neutral h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            dataframe_pos (unicode): pandas.Dataframe converted to html table
            using pandas.DataFrame.to_html method. Displayed in
            <div id=dataframe_pos></div> if is not <None> (default).
            dataframe_neg (unicode): pandas.Dataframe converted to html table
            using pandas.DataFrame.to_html method. Displayed in
            <div id=dataframe_neg></div> if is not <None> (default).
            dataframe_neu (unicode): pandas.Dataframe converted to html table
            using pandas.DataFrame.to_html method. Displayed in
            <div id=dataframe_neu></div> if is not <None> (default).
        """
        curr_time = time.localtime()
        time_stamp = "{0}.{1}.{2}, {3}:{4}:{5}".format(curr_time.tm_year,
                                                       curr_time.tm_mon,
                                                       curr_time.tm_mday,
                                                       curr_time.tm_hour,
                                                       curr_time.tm_min,
                                                       curr_time.tm_sec)
        template_Vars = {"time_stamp": time_stamp,
                         "name": name,
                         "filters_pos": filters_pos,
                         "filters_neg": filters_neg,
                         "filters_neu": filters_neu,
                         "definitions": self.definitions,
                         "num_prop_res": num_prop_res,
                         "num_prop_perm": num_prop_perm,
                         "histogram_bins": histogram_bins,
                         "e_value": e_value,
                         "histogram_gis_pos": histogram_gis_pos,
                         "bivar_pos": bivar_pos,
                         "lin_regr_pos": lin_regr_pos,
                         "histogram_gis_neg": histogram_gis_neg,
                         "bivar_neg": bivar_neg,
                         "lin_regr_neg": lin_regr_neg,
                         "histogram_gis_neu": histogram_gis_neu,
                         "bivar_neu": bivar_neu,
                         "lin_regr_neu": lin_regr_neu,
                         "dataframe_pos": dataframe_pos,
                         "dataframe_neg": dataframe_neg,
                         "dataframe_neu": dataframe_neu,
                         "results_type": results_type,
                         "skip_perm_res": skip_perm_res}
        self.template_rendered = self.template.render(template_Vars)

    def save_template(self,
                      out_file_name):
        """Save rendered template to file.

        Args:
            out_file_name (str): name for file to be saved
        """
        with open("{0}.html".format(out_file_name), "w") as fout:
            fout.write(self.template_rendered)

    def print_template(self):
        """Print template as it is rendered to str(). Just for debugging. Will
        be removed.
        """
        print self.template.render()
