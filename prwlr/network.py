# -*- coding: utf-8 -*-


from __future__ import print_function
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


class Ortho_Network(object):
    """Calculates and holds data about interactions in form of network,
    exportable to other software (e.g. Cytoscape) or drawable by matplotlib.

    Attribs:
        inter_df (pandas.DataFrame): DataFrame containing genetic interactions
        nwrk (networkx.Graph): network created upon Ortho_Network.inter_df
    """
    def __init__(self,
                 inter_df):
        self.inter_df = inter_df
        self.nwrk = None
        self.sub_nwrk = None

    def create_nwrk(self,
                    nodes_cols,
                    attribs_cols):
        """Return Ortho_Network.nwrk upon pandas.DataFrame.

        Parameters
        -------
        nodes_cols: list
            Columns to take as nodes.
        attribs_cols: list
            Columns to take as attributes.

        Returns
        -------
        P_CRAWLER.Ortho_Network.nwrk
            Interactions-based network. networkx.classes.graph.Graph derivate.
        """
        self.nwrk = nx.from_pandas_dataframe(self.inter_df,
                                             nodes_cols[0],
                                             nodes_cols[1],
                                             attribs_cols)

    def get_subgrps(self):
        self.sub_nwrk = [i for i in nx.connected_component_subgraphs(self.nwrk)]

    def write_nwrk(self,
                   out_file_name,
                   out_file_format):
        """Write Ortho_Network.nwrk to file readable to other software.

        Args:
            out_file_name (str): file name to save as
            out_file_format (str): file format to save as. Available formats are:
            graphml, gefx, gml, json
        """
        if out_file_format.lower() == "graphml":
            nx.write_graphml(self.nwrk, out_file_name)
        elif out_file_format.lower() == "gefx":
            nx.write_gexf(self.nwrk, out_file_name)
        elif out_file_format.lower() == "gml":
            nx.write_gml(self.nwrk, out_file_name)
        elif out_file_format.lower() == "json":
            with open("{0}.{1}".format(out_file_name, out_file_format), "w") as fout:
                fout.write(json.dumps(json_graph.node_link_data(self.nwrk)))

    def draw_nwrk(self,
                  width=20,
                  height=20,
                  dpi=None,
                  node_size=5,
                  save_2_file=False,
                  out_file_name="network.png",
                  sub_nwrk=False):
        """Return matplotlib.pyplot.figure of Ortho_Network.nwrk and/or write it to
        <*.png> file.

        Args:
            width (int): figure width in inches. Set as speciefied in
            matplotlibrc file when <None>. Default: <20>
            height (int): figure height in inches. Set as speciefied in
            matplotlibrc file when <None>. Default: <20>
            dpi (int): figure resolution. Set as speciefied in
            matplotlibrc file when <None>. Default: <None>
            node_size (int): size of the nodes. Default: <5>
            save_2_file (bool): write to file when <True>. Default: <False>
            out_file_name (str): file name to save as

        """
        if sub_nwrk is True:
            nwrk = self.sub_nwrk
        else:
            nwrk = self.nwrk
        plt.figure(figsize=(width, height))
        nx.draw_networkx(nwrk,
                         node_size=node_size,
                         node_color="r",
                         node_alpha=0.4,
                         with_labels=False)
        if save_2_file is True:
            plt.savefig(out_file_name,
                        dpi=dpi)
        else:
            pass
