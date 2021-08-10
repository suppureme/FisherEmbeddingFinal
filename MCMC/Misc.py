class Misc:

    BLANK = "<BLANK>"

    def __init__(self):
        pass


    @staticmethod
    def apply_if_not_present( df, cls, to_delete ):
        try:
            idx = df.columns.get_loc( cls.describe() )
        except:
            print( "Could not find " + str( cls.describe() ))
            df = cls.apply(df)
            to_delete.append( cls.describe() )
        return [ df, to_delete ]

    @staticmethod
    def roc_pct(row, horizon, feature ):
        change = row[ feature ] - row[ feature + "_T-" + str( horizon ) ]
        change_pct = change/row[feature + "_T-" + str( horizon )]
        return change_pct

    @staticmethod
    def change(row, horizon, feature):
        chg = row[feature] - row[feature + "_T-" + str(horizon)]
        return chg

    @staticmethod
    def rsi( row, rma_adv, rma_dec, sum_n_adv, sum_n_dec ):
        sum_n_adv_v = abs(row[sum_n_adv ])
        sum_n_dec_v = abs(row[sum_n_dec])

        rma_adv_v = abs( row[rma_adv])
        rma_dec_v = abs( row[rma_dec])

        mean_adv_v = rma_adv_v
        mean_dec_v = rma_dec_v

        if mean_dec_v == 0:
            ratio = 0
        else:
            ratio = 100/(1+(mean_adv_v/mean_dec_v))

        r = 100 - ratio

        return r