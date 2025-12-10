import eyepy as ep
import imageio.v3 as imageio

ev = ep.import_heyex_vol("data/350436.vol")
data = ev.data

imageio.imwrite("data/350436.tif", data)
