import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface CandlestickData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface D3CandlestickChartProps {
  data: CandlestickData[];
  width?: number;
  height?: number;
  title?: string;
  style?: React.CSSProperties;
}

export const D3CandlestickChart: React.FC<D3CandlestickChartProps> = ({
  data,
  width = 800,
  height = 400,
  title,
  style = {},
}) => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!data || data.length === 0) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Margins
    const margin = { top: 40, right: 60, bottom: 80, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const volumeHeight = innerHeight * 0.2; // 20% for volume
    const priceHeight = innerHeight * 0.8; // 80% for price

    // Parse dates
    const parseDate = d3.timeParse('%Y-%m-%d');
    const parsedData = data.map(d => ({
      ...d,
      date: parseDate(d.date) || new Date()
    }));

    // Scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(parsedData, d => d.date) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3
      .scaleLinear()
      .domain([
        d3.min(parsedData, d => d.low) ?? 0,
        d3.max(parsedData, d => d.high) ?? 1
      ])
      .nice()
      .range([priceHeight, 0]);

    const volumeScale = d3
      .scaleLinear()
      .domain([0, d3.max(parsedData, d => d.volume) ?? 1])
      .range([innerHeight, priceHeight + 20]);

    // Axes
    const xAxis = d3.axisBottom(xScale).tickFormat(d3.timeFormat('%m/%d'));
    const yAxis = d3.axisLeft(yScale);
    const volumeAxis = d3.axisRight(volumeScale).ticks(3);

    // Main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Gridlines
    g.append('g')
      .attr('class', 'grid')
      .selectAll('line')
      .data(yScale.ticks(8))
      .join('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', d => yScale(d))
      .attr('y2', d => yScale(d))
      .attr('stroke', '#eee')
      .attr('stroke-width', 0.5);

    // Tooltip
    const tooltip = d3
      .select(ref.current?.parentElement)
      .selectAll('.d3-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'd3-tooltip')
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('background', 'rgba(255,255,255,0.95)')
      .style('border', '1px solid #bbb')
      .style('border-radius', '4px')
      .style('padding', '8px 12px')
      .style('font-size', '12px')
      .style('color', '#222')
      .style('box-shadow', '0 2px 8px rgba(0,0,0,0.1)')
      .style('display', 'none');

    // Volume bars
    g.selectAll('.volume-bar')
      .data(parsedData)
      .join('rect')
      .attr('class', 'volume-bar')
      .attr('x', d => xScale(d.date) - 2)
      .attr('y', d => volumeScale(d.volume))
      .attr('width', 4)
      .attr('height', d => innerHeight - volumeScale(d.volume))
      .attr('fill', d => d.close > d.open ? '#26a69a' : '#ef5350')
      .attr('opacity', 0.3);

    // Candlestick wicks
    g.selectAll('.wick')
      .data(parsedData)
      .join('line')
      .attr('class', 'wick')
      .attr('x1', d => xScale(d.date))
      .attr('x2', d => xScale(d.date))
      .attr('y1', d => yScale(d.high))
      .attr('y2', d => yScale(d.low))
      .attr('stroke', d => d.close > d.open ? '#26a69a' : '#ef5350')
      .attr('stroke-width', 1);

    // Candlestick bodies
    g.selectAll('.candle')
      .data(parsedData)
      .join('rect')
      .attr('class', 'candle')
      .attr('x', d => xScale(d.date) - 3)
      .attr('y', d => yScale(Math.max(d.open, d.close)))
      .attr('width', 6)
      .attr('height', d => Math.abs(yScale(d.open) - yScale(d.close)) || 1)
      .attr('fill', d => d.close > d.open ? '#26a69a' : '#ef5350')
      .attr('stroke', d => d.close > d.open ? '#26a69a' : '#ef5350')
      .attr('stroke-width', 1)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 0.8);
        tooltip
          .style('display', 'block')
          .html(`
            <strong>Date:</strong> ${d.date.toLocaleDateString()}<br/>
            <strong>Open:</strong> $${d.open.toFixed(2)}<br/>
            <strong>High:</strong> $${d.high.toFixed(2)}<br/>
            <strong>Low:</strong> $${d.low.toFixed(2)}<br/>
            <strong>Close:</strong> $${d.close.toFixed(2)}<br/>
            <strong>Volume:</strong> ${d.volume.toLocaleString()}
          `)
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 28 + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 1);
        tooltip.style('display', 'none');
      });

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .call(g => g.selectAll('.domain').attr('stroke', '#bbb'))
      .call(g => g.selectAll('text').attr('font-size', 12).attr('fill', '#888'));

    g.append('g')
      .call(yAxis)
      .call(g => g.selectAll('.domain').attr('stroke', '#bbb'))
      .call(g => g.selectAll('text').attr('font-size', 12).attr('fill', '#888'));

    g.append('g')
      .attr('transform', `translate(${innerWidth},0)`)
      .call(volumeAxis)
      .call(g => g.selectAll('.domain').attr('stroke', '#bbb'))
      .call(g => g.selectAll('text').attr('font-size', 10).attr('fill', '#888'));

    // Title
    if (title) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', 18)
        .attr('font-family', 'Georgia, serif')
        .attr('font-weight', 'bold')
        .attr('fill', '#222')
        .text(title);
    }

    // Labels
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', height - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', 14)
      .attr('fill', '#888')
      .text('Date');

    svg
      .append('text')
      .attr('transform', `rotate(-90)`)
      .attr('x', -height / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 14)
      .attr('fill', '#888')
      .text('Price');

    svg
      .append('text')
      .attr('transform', `rotate(-90)`)
      .attr('x', -height / 2)
      .attr('y', width - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', '#888')
      .text('Volume');

  }, [data, width, height, title]);

  return (
    <div style={{ position: 'relative', width, height, ...style }}>
      <svg ref={ref} width={width} height={height} />
    </div>
  );
}; 